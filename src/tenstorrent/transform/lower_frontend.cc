/*!
 * \file tl/tenstorrent/transform/lower_frontend.cc
 * \brief Lower Tenstorrent frontend launch, topology, and allocation metadata.
 */

#include "../../op/builtin.h"
#include "../op/builtin.h"

#include <tvm/ffi/extra/json.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace transform {

using namespace ffi;
using namespace tirx;

namespace {

constexpr const char *kAllocBufferMetadata = "tl.tt.alloc_buffer_metadata";
constexpr const char *kDfbBlockCount = "tt.dfb_block_count";
constexpr const char *kTileShape = "tt.tile_shape";
constexpr const char *kTensorBacked = "tt.tensor_backed";
constexpr const char *kForeachSrc = "tl.tt.foreach_src";
constexpr const char *kForeachDst = "tl.tt.foreach_dst";

int64_t RequireIntImm(const PrimExpr &value, const std::string &context) {
  if (const auto *imm = value.as<IntImmNode>()) {
    return imm->value;
  }
  TVM_FFI_THROW(ValueError) << context << " must be a compile-time integer";
  return 0;
}

int64_t RequireAnnotationInt(const Any &value, const std::string &context) {
  if (auto expr = value.try_cast<PrimExpr>()) {
    if (expr.value().dtype().is_bool()) {
      TVM_FFI_THROW(ValueError)
          << context << " must be a compile-time integer, not bool";
    }
    return RequireIntImm(expr.value(), context);
  }
  if (value.GetTypeKey() == "bool") {
    TVM_FFI_THROW(ValueError)
        << context << " must be a compile-time integer, not bool";
  }
  if (auto integer = value.try_cast<int64_t>()) {
    return integer.value();
  }
  TVM_FFI_THROW(ValueError)
      << context << " must be a compile-time integer, got "
      << value.GetTypeKey();
  return 0;
}

class TenstorrentLaunchValidator : public StmtVisitor {
public:
  void Validate(const Stmt &body) {
    VisitStmt(body);
    if (block_x_ != 1 || block_y_ != 1 || block_z_ != 0) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent kernel launch requires exactly a static 2-D Core "
             "grid using blockIdx.x and blockIdx.y (and no blockIdx.z)";
    }
    if (thread_x_ != 1 || thread_y_ != 1 || thread_z_ != 1) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent kernel launch requires threadIdx.x, threadIdx.y, "
             "and threadIdx.z extent to be 1";
    }
  }

private:
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined()) {
      const std::string tag = op->thread_binding.value()->thread_tag;
      const auto *min_imm = op->min.as<IntImmNode>();
      const auto *extent_imm = op->extent.as<IntImmNode>();
      if (tag.rfind("blockIdx.", 0) == 0 &&
          (min_imm == nullptr || extent_imm == nullptr)) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent kernel launch requires exactly a static 2-D Core "
               "grid using blockIdx.x and blockIdx.y";
      }
      if (tag.rfind("threadIdx.", 0) == 0 &&
          (min_imm == nullptr || extent_imm == nullptr)) {
        TVM_FFI_THROW(ValueError) << "Tenstorrent kernel launch requires "
                                  << tag << " extent to be 1";
      }
      if (min_imm == nullptr || extent_imm == nullptr) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent kernel launch thread binding `" << tag
            << "` must have compile-time minimum and extent";
      }
      const int64_t min = min_imm->value;
      const int64_t extent = extent_imm->value;
      if (min != 0 || extent <= 0) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent kernel launch " << tag
            << " must have compile-time minimum 0 and positive extent";
      }
      if (tag == "blockIdx.x") {
        ++block_x_;
      } else if (tag == "blockIdx.y") {
        ++block_y_;
      } else if (tag == "blockIdx.z") {
        ++block_z_;
      } else if (tag == "threadIdx.x") {
        ++thread_x_;
        ValidateUnitThread(tag, extent);
      } else if (tag == "threadIdx.y") {
        ++thread_y_;
        ValidateUnitThread(tag, extent);
      } else if (tag == "threadIdx.z") {
        ++thread_z_;
        ValidateUnitThread(tag, extent);
      } else {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent kernel launch does not support thread binding `"
            << tag << "`";
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  static void ValidateUnitThread(const std::string &tag, int64_t extent) {
    if (extent != 1) {
      TVM_FFI_THROW(ValueError) << "Tenstorrent kernel launch requires " << tag
                                << " extent to be 1, got " << extent;
    }
  }

  int block_x_{0};
  int block_y_{0};
  int block_z_{0};
  int thread_x_{0};
  int thread_y_{0};
  int thread_z_{0};
};

struct CoreCoord {
  int64_t x;
  int64_t y;
};

struct PipeDescriptor {
  CoreCoord src;
  bool collective{false};
  CoreCoord dst_begin;
  CoreCoord dst_end;
};

struct PipeNetDescriptor {
  int64_t id;
  std::string kind;
  std::vector<PipeDescriptor> pipes;
};

Any RequireJsonField(const json::Object &object, const char *field,
                     const std::string &context) {
  for (const auto &[key, value] : object) {
    if (auto string_key = key.try_cast<String>()) {
      if (string_key.value() == field) {
        return value;
      }
    }
  }
  TVM_FFI_THROW(ValueError)
      << context << " is missing required field `" << field << "`";
  return Any();
}

json::Object RequireJsonObject(const Any &value, const std::string &context) {
  if (auto object = value.try_cast<json::Object>()) {
    return object.value();
  }
  TVM_FFI_THROW(ValueError) << context << " must be a JSON object";
  return json::Object();
}

json::Array RequireJsonArray(const Any &value, const std::string &context) {
  if (auto array = value.try_cast<json::Array>()) {
    return array.value();
  }
  TVM_FFI_THROW(ValueError) << context << " must be a JSON array";
  return json::Array();
}

int64_t RequireJsonInt(const Any &value, const std::string &context) {
  if (auto integer = value.try_cast<int64_t>()) {
    return integer.value();
  }
  TVM_FFI_THROW(ValueError) << context << " must be a JSON integer";
  return 0;
}

std::string RequireJsonString(const Any &value, const std::string &context) {
  if (auto string = value.try_cast<String>()) {
    return string.value();
  }
  TVM_FFI_THROW(ValueError) << context << " must be a JSON string";
  return std::string();
}

CoreCoord ParseCoord(const Any &value, const std::string &context) {
  json::Array array = RequireJsonArray(value, context);
  if (array.size() != 2) {
    TVM_FFI_THROW(ValueError) << context << " must contain exactly [x, y]";
  }
  CoreCoord result{RequireJsonInt(array[0], context + "[0]"),
                   RequireJsonInt(array[1], context + "[1]")};
  if (result.x < 0 || result.y < 0) {
    TVM_FFI_THROW(ValueError) << context << " coordinates must be non-negative";
  }
  return result;
}

class TenstorrentFrontendLowerer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    if (func->GetAttr<Map<Var, Map<String, Any>>>(kAllocBufferMetadata)) {
      TVM_FFI_THROW(ValueError)
          << "PrimFunc already has transient `" << kAllocBufferMetadata
          << "`; run LowerTenstorrentBufferAllocations before lowering new "
             "frontend annotations";
    }
    TenstorrentFrontendLowerer lowerer;
    func.CopyOnWrite()->body = lowerer(func->body);
    if (!lowerer.buffer_metadata_.empty()) {
      func = WithAttr(std::move(func), kAllocBufferMetadata,
                      lowerer.buffer_metadata_);
    }
    return func;
  }

private:
  enum class PipeSide { kSrc, kDst };

  struct PipeContext {
    Var loop_var;
    PipeDescriptor pipe;
    int64_t index;
    PipeSide side;
  };

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent) {
      return StmtExprMutator::VisitStmt_(op);
    }
    IterVar iter_var = Downcast<IterVar>(op->node);
    const std::string tag = iter_var->thread_tag;
    std::optional<Var> old_var;
    std::optional<int64_t> old_extent;
    if (tag == "blockIdx.x") {
      old_var = core_x_;
      old_extent = grid_x_;
      core_x_ = iter_var->var;
      grid_x_ = RequireIntImm(op->value, "Tenstorrent Core grid x extent");
    } else if (tag == "blockIdx.y") {
      old_var = core_y_;
      old_extent = grid_y_;
      core_y_ = iter_var->var;
      grid_y_ = RequireIntImm(op->value, "Tenstorrent Core grid y extent");
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    if (tag == "blockIdx.x") {
      core_x_ = old_var;
      grid_x_ = old_extent;
    } else if (tag == "blockIdx.y") {
      core_y_ = old_var;
      grid_y_ = old_extent;
    }
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto src_annotation = op->annotations.Get(kForeachSrc);
    auto dst_annotation = op->annotations.Get(kForeachDst);
    if (!src_annotation && !dst_annotation) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (src_annotation && dst_annotation) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent foreach loop cannot have both `" << kForeachSrc
          << "` and `" << kForeachDst << "` annotations";
    }
    RequireCoreContext("T.tt.foreach_src/dst");
    Any annotation =
        src_annotation ? src_annotation.value() : dst_annotation.value();
    auto descriptor_string = annotation.try_cast<String>();
    if (!descriptor_string) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent foreach annotation must contain a serialized "
             "PipeNet string";
    }
    PipeNetDescriptor net = ParsePipeNet(descriptor_string.value());
    Array<Stmt> expanded;
    for (size_t i = 0; i < net.pipes.size(); ++i) {
      const PipeDescriptor &pipe = net.pipes[i];
      contexts_.push_back(
          PipeContext{op->loop_var, pipe, static_cast<int64_t>(i),
                      src_annotation ? PipeSide::kSrc : PipeSide::kDst});
      Stmt body = VisitStmt(op->body);
      contexts_.pop_back();
      PrimExpr active = src_annotation ? IsSource(pipe) : IsDestination(pipe);
      expanded.push_back(IfThenElse(std::move(active), std::move(body)));
    }
    return SeqStmt::Flatten(std::move(expanded));
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto annotation = block->annotations.Get(tl::attr::kAllocBufferAnnotations);
    if (!annotation) {
      return block;
    }
    auto metadata = annotation.value().try_cast<Map<Var, Map<String, Any>>>();
    if (!metadata) {
      TVM_FFI_THROW(ValueError) << "`" << tl::attr::kAllocBufferAnnotations
                                << "` must be Map<Var, Map<String, Any>>";
    }
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>
        local_buffers;
    for (const Buffer &buffer : block->alloc_buffers) {
      local_buffers.emplace(buffer->data, buffer);
    }
    for (const auto &[var, values] : metadata.value()) {
      auto buffer_it = local_buffers.find(var);
      if (buffer_it == local_buffers.end()) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent allocation metadata for `" << var->name_hint
            << "` does not refer to an alloc_buffer in the annotated SBlock";
      }
      if (buffer_metadata_.count(var)) {
        TVM_FFI_THROW(ValueError)
            << "Duplicate Tenstorrent allocation metadata for buffer `"
            << var->name_hint << "`";
      }
      buffer_metadata_.Set(var,
                           NormalizeBufferMetadata(buffer_it->second, values));
    }
    block.CopyOnWrite()->annotations.erase(tl::attr::kAllocBufferAnnotations);
    return block;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
      if (var.same_as(it->loop_var)) {
        return IntImm(var.dtype(), it->index);
      }
    }
    return var;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tenstorrent::is_src()) ||
        op->op.same_as(tenstorrent::is_dst()) ||
        op->op.same_as(tenstorrent::is_active())) {
      RequireCoreContext("T.tt topology predicate");
      if (op->args.size() != 1) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent topology predicate expects one PipeNet descriptor";
      }
      const auto *descriptor = op->args[0].as<StringImmNode>();
      if (descriptor == nullptr) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent topology predicate requires a serialized PipeNet "
               "StringImm";
      }
      PipeNetDescriptor net = ParsePipeNet(descriptor->value);
      PrimExpr result = Bool(false);
      for (const PipeDescriptor &pipe : net.pipes) {
        PrimExpr member;
        if (op->op.same_as(tenstorrent::is_src())) {
          member = IsSource(pipe);
        } else if (op->op.same_as(tenstorrent::is_dst())) {
          member = IsDestination(pipe);
        } else {
          member = Or(IsSource(pipe), IsDestination(pipe));
        }
        result = Or(std::move(result), std::move(member));
      }
      return result;
    }

    if (op->op.same_as(tenstorrent::pipe_src()) ||
        op->op.same_as(tenstorrent::pipe_dst()) ||
        op->op.same_as(tenstorrent::pipe_dst_range())) {
      return LowerPipeAccessor(op);
    }

    if (op->op.same_as(tenstorrent::pipe_send()) ||
        op->op.same_as(tenstorrent::pipe_recv())) {
      return LowerPipeTransfer(op);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Map<String, Any> NormalizeBufferMetadata(const Buffer &buffer,
                                           const Map<String, Any> &values) {
    Map<String, Any> result;
    bool has_tt_metadata = false;
    for (const auto &[key, value] : values) {
      const std::string key_string = key;
      if (key_string == kTensorBacked) {
        TVM_FFI_THROW(ValueError)
            << "`" << kTensorBacked
            << "` is not supported by the Phase 1 Tenstorrent allocation "
               "lowering";
      }
      if (key_string.rfind("tt.", 0) == 0 && key_string != kDfbBlockCount &&
          key_string != kTileShape) {
        TVM_FFI_THROW(ValueError)
            << "Unsupported Tenstorrent alloc_shared annotation `" << key
            << "`";
      }
      if (key_string == kDfbBlockCount || key_string == kTileShape) {
        has_tt_metadata = true;
      } else {
        result.Set(key, value);
      }
    }
    if (!has_tt_metadata) {
      return result;
    }
    const std::string scope = buffer.scope();
    if (scope != "shared" && scope != "shared.dyn") {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent DFB metadata requires a shared or shared.dyn "
             "buffer, got scope `"
          << scope << "`";
    }
    if (buffer->shape.size() != 2) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent DFB metadata requires a 2-D shared buffer, got rank "
          << buffer->shape.size();
    }

    if (auto block_count_value = values.Get(kDfbBlockCount)) {
      int64_t block_count =
          RequireAnnotationInt(block_count_value.value(), kDfbBlockCount);
      if (block_count < 1 || block_count > 32) {
        TVM_FFI_THROW(ValueError)
            << "`" << kDfbBlockCount << "` must be in [1, 32], got "
            << block_count;
      }
      result.Set(kDfbBlockCount, IntImm(DataType::Int(32), block_count));
    }

    Array<PrimExpr> tile_shape{IntImm(DataType::Int(32), 32),
                               IntImm(DataType::Int(32), 32)};
    if (auto tile_shape_value = values.Get(kTileShape)) {
      auto shape = tile_shape_value.value().try_cast<Array<PrimExpr>>();
      if (!shape || shape->size() != 2) {
        TVM_FFI_THROW(ValueError)
            << "`" << kTileShape << "` must be a pair of compile-time integers";
      }
      tile_shape = shape.value();
      for (size_t i = 0; i < 2; ++i) {
        if (RequireIntImm(tile_shape[i], std::string(kTileShape) + " axis " +
                                             std::to_string(i)) != 32) {
          TVM_FFI_THROW(ValueError)
              << "Phase 1 only supports `" << kTileShape << "=(32, 32)`";
        }
      }
    }
    for (size_t i = 0; i < 2; ++i) {
      int64_t extent =
          RequireIntImm(buffer->shape[i], "Tenstorrent DFB buffer shape axis " +
                                              std::to_string(i));
      int64_t tile_extent = RequireIntImm(tile_shape[i], kTileShape);
      if (extent % tile_extent != 0) {
        TVM_FFI_THROW(ValueError)
            << "Shared buffer shape axis " << i << " (" << extent
            << ") must be divisible by `" << kTileShape << "` axis " << i
            << " (" << tile_extent << ")";
      }
    }
    result.Set(kTileShape, tile_shape);
    return result;
  }

  PipeNetDescriptor ParsePipeNet(const String &encoded) {
    String parse_error;
    json::Value parsed = json::Parse(encoded, &parse_error);
    if (parsed == nullptr) {
      TVM_FFI_THROW(ValueError)
          << "Invalid serialized Tenstorrent PipeNet: " << parse_error;
    }
    json::Object root = RequireJsonObject(parsed, "Tenstorrent PipeNet");
    PipeNetDescriptor result;
    result.id =
        RequireJsonInt(RequireJsonField(root, "id", "Tenstorrent PipeNet"),
                       "Tenstorrent PipeNet.id");
    result.kind =
        RequireJsonString(RequireJsonField(root, "kind", "Tenstorrent PipeNet"),
                          "Tenstorrent PipeNet.kind");
    if (result.kind != "point_to_point" && result.kind != "collective") {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent PipeNet.kind must be `point_to_point` or "
             "`collective`, got `"
          << result.kind << "`";
    }
    const std::string encoded_string = encoded;
    auto known = pipenet_encodings_.find(result.id);
    if (known != pipenet_encodings_.end() && known->second != encoded_string) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent PipeNet id " << result.id
          << " is used with inconsistent serialized descriptors";
    }
    pipenet_encodings_[result.id] = encoded_string;

    json::Array pipes =
        RequireJsonArray(RequireJsonField(root, "pipes", "Tenstorrent PipeNet"),
                         "Tenstorrent PipeNet.pipes");
    if (pipes.empty()) {
      TVM_FFI_THROW(ValueError)
          << "Tenstorrent PipeNet requires at least one Pipe";
    }
    for (size_t i = 0; i < pipes.size(); ++i) {
      const std::string context =
          "Tenstorrent PipeNet.pipes[" + std::to_string(i) + "]";
      json::Object pipe_object = RequireJsonObject(pipes[i], context);
      PipeDescriptor pipe;
      pipe.src = ParseCoord(RequireJsonField(pipe_object, "src", context),
                            context + ".src");
      Any dst = RequireJsonField(pipe_object, "dst", context);
      if (result.kind == "point_to_point") {
        pipe.dst_begin = ParseCoord(dst, context + ".dst");
        pipe.dst_end = CoreCoord{pipe.dst_begin.x + 1, pipe.dst_begin.y + 1};
      } else {
        pipe.collective = true;
        json::Object range = RequireJsonObject(dst, context + ".dst");
        pipe.dst_begin =
            ParseCoord(RequireJsonField(range, "begin", context + ".dst"),
                       context + ".dst.begin");
        pipe.dst_end =
            ParseCoord(RequireJsonField(range, "end", context + ".dst"),
                       context + ".dst.end");
        if (pipe.dst_begin.x >= pipe.dst_end.x ||
            pipe.dst_begin.y >= pipe.dst_end.y) {
          TVM_FFI_THROW(ValueError)
              << context << ".dst must be a non-empty half-open Core range";
        }
      }
      ValidateInGrid(pipe.src, context + ".src", false);
      ValidateInGrid(pipe.dst_begin, context + ".dst.begin", false);
      ValidateInGrid(pipe.dst_end, context + ".dst.end", true);
      result.pipes.push_back(pipe);
    }
    return result;
  }

  void ValidateInGrid(const CoreCoord &coord, const std::string &context,
                      bool allow_end) const {
    RequireCoreContext(context);
    const bool x_valid =
        allow_end ? coord.x <= grid_x_.value() : coord.x < grid_x_.value();
    const bool y_valid =
        allow_end ? coord.y <= grid_y_.value() : coord.y < grid_y_.value();
    if (!x_valid || !y_valid) {
      TVM_FFI_THROW(ValueError)
          << context << " coordinate (" << coord.x << ", " << coord.y
          << ") is outside Tenstorrent Core grid (" << grid_x_.value() << ", "
          << grid_y_.value() << ")";
    }
  }

  void RequireCoreContext(const std::string &context) const {
    if (!core_x_ || !core_y_ || !grid_x_ || !grid_y_) {
      TVM_FFI_THROW(ValueError)
          << context
          << " must be inside a materialized 2-D Tenstorrent Core grid; run "
             "MaterializeKernelLaunch before "
             "LowerTenstorrentFrontendAnnotations";
    }
  }

  PrimExpr IsSource(const PipeDescriptor &pipe) const {
    return And(core_x_.value() == IntImm(DataType::Int(32), pipe.src.x),
               core_y_.value() == IntImm(DataType::Int(32), pipe.src.y));
  }

  PrimExpr IsDestination(const PipeDescriptor &pipe) const {
    if (!pipe.collective) {
      return And(core_x_.value() == IntImm(DataType::Int(32), pipe.dst_begin.x),
                 core_y_.value() ==
                     IntImm(DataType::Int(32), pipe.dst_begin.y));
    }
    return And(
        And(core_x_.value() >= IntImm(DataType::Int(32), pipe.dst_begin.x),
            core_x_.value() < IntImm(DataType::Int(32), pipe.dst_end.x)),
        And(core_y_.value() >= IntImm(DataType::Int(32), pipe.dst_begin.y),
            core_y_.value() < IntImm(DataType::Int(32), pipe.dst_end.y)));
  }

  const PipeContext &RequirePipeContext(const PrimExpr &selected,
                                        const std::string &op_name) const {
    const auto *var_node = selected.as<VarNode>();
    if (var_node == nullptr) {
      TVM_FFI_THROW(ValueError)
          << op_name
          << " requires the loop Var selected by T.tt.foreach_src/dst";
    }
    Var var = GetRef<Var>(var_node);
    for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
      if (var.same_as(it->loop_var)) {
        return *it;
      }
    }
    TVM_FFI_THROW(ValueError)
        << op_name
        << " selected PipeRef can only be used inside its "
           "T.tt.foreach_src/dst region";
    return contexts_.front();
  }

  PrimExpr LowerPipeAccessor(const CallNode *op) {
    const std::string op_name =
        op->op.same_as(tenstorrent::pipe_src())   ? "T.tt.pipe_src"
        : op->op.same_as(tenstorrent::pipe_dst()) ? "T.tt.pipe_dst"
                                                  : "T.tt.pipe_dst_range";
    if (op->args.empty()) {
      TVM_FFI_THROW(ValueError)
          << op_name << " is missing its selected PipeRef";
    }
    const PipeContext &context = RequirePipeContext(op->args[0], op_name);
    if (op->op.same_as(tenstorrent::pipe_src()) ||
        op->op.same_as(tenstorrent::pipe_dst())) {
      if (op->args.size() != 2) {
        TVM_FFI_THROW(ValueError) << op_name << " expects a dimension selector";
      }
      if (op->op.same_as(tenstorrent::pipe_dst()) && context.pipe.collective) {
        TVM_FFI_THROW(ValueError)
            << "T.tt.pipe_dst requires a point-to-point PipeNet";
      }
      int64_t dimension = RequireIntImm(op->args[1], op_name + " dimension");
      if (dimension != 0 && dimension != 1) {
        TVM_FFI_THROW(ValueError) << op_name << " dimension must be 0 or 1";
      }
      CoreCoord coord = op->op.same_as(tenstorrent::pipe_src())
                            ? context.pipe.src
                            : context.pipe.dst_begin;
      return IntImm(op->dtype, dimension == 0 ? coord.x : coord.y);
    }
    if (op->args.size() != 3 || !context.pipe.collective) {
      TVM_FFI_THROW(ValueError)
          << "T.tt.pipe_dst_range requires a collective PipeNet and endpoint "
             "and dimension selectors";
    }
    int64_t endpoint =
        RequireIntImm(op->args[1], "T.tt.pipe_dst_range endpoint");
    int64_t dimension =
        RequireIntImm(op->args[2], "T.tt.pipe_dst_range dimension");
    if ((endpoint != 0 && endpoint != 1) ||
        (dimension != 0 && dimension != 1)) {
      TVM_FFI_THROW(ValueError)
          << "T.tt.pipe_dst_range selectors must be 0 or 1";
    }
    CoreCoord coord =
        endpoint == 0 ? context.pipe.dst_begin : context.pipe.dst_end;
    return IntImm(op->dtype, dimension == 0 ? coord.x : coord.y);
  }

  PrimExpr LowerPipeTransfer(const CallNode *op) {
    const bool is_send = op->op.same_as(tenstorrent::pipe_send());
    const std::string op_name = is_send ? "T.copy send" : "T.copy receive";
    if (op->args.size() != 2) {
      TVM_FFI_THROW(ValueError)
          << op_name << " expects a payload region and PipeRef";
    }
    const PrimExpr &selected = is_send ? op->args[1] : op->args[0];
    const PrimExpr &region = is_send ? op->args[0] : op->args[1];
    const PipeContext &context = RequirePipeContext(selected, op_name);
    if ((is_send && context.side != PipeSide::kSrc) ||
        (!is_send && context.side != PipeSide::kDst)) {
      TVM_FFI_THROW(ValueError)
          << op_name << " uses a PipeRef from the wrong foreach side";
    }
    PrimExpr lowered_region = VisitExpr(region);
    if (is_send) {
      return Call(op->dtype, tenstorrent::noc_send(),
                  {lowered_region,
                   IntImm(DataType::Int(32), context.pipe.dst_begin.x),
                   IntImm(DataType::Int(32), context.pipe.dst_begin.y),
                   IntImm(DataType::Int(32), context.pipe.dst_end.x),
                   IntImm(DataType::Int(32), context.pipe.dst_end.y)},
                  op->annotations, op->span);
    }
    return Call(op->dtype, tenstorrent::noc_recv(),
                {lowered_region, IntImm(DataType::Int(32), context.pipe.src.x),
                 IntImm(DataType::Int(32), context.pipe.src.y)},
                op->annotations, op->span);
  }

  Map<Var, Map<String, Any>> buffer_metadata_;
  std::optional<Var> core_x_;
  std::optional<Var> core_y_;
  std::optional<int64_t> grid_x_;
  std::optional<int64_t> grid_y_;
  std::vector<PipeContext> contexts_;
  std::unordered_map<int64_t, std::string> pipenet_encodings_;
};

class TenstorrentBufferAllocationLowerer : public StmtExprMutator {
public:
  explicit TenstorrentBufferAllocationLowerer(
      Map<Var, Map<String, Any>> metadata)
      : metadata_(std::move(metadata)) {}

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    auto metadata = metadata_.Get(op->buffer->data);
    if (!metadata) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (!consumed_.insert(op->buffer->data).second) {
      TVM_FFI_THROW(ValueError) << "Tenstorrent allocation metadata for `"
                                << op->buffer->data->name_hint
                                << "` matched more than one AllocBuffer";
    }
    Map<String, Any> annotations = op->annotations;
    for (const auto &[key, value] : metadata.value()) {
      if (annotations.count(key)) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent allocation metadata key `" << key
            << "` already exists on AllocBuffer `" << op->buffer->name << "`";
      }
      annotations.Set(key, value);
    }
    return AllocBuffer(op->buffer, std::move(annotations), op->span);
  }

  void VerifyConsumed() const {
    for (const auto &[var, _] : metadata_) {
      if (!consumed_.count(var)) {
        TVM_FFI_THROW(ValueError)
            << "Tenstorrent allocation metadata for `" << var->name_hint
            << "` did not match an AllocBuffer; ensure "
               "LowerTenstorrentBufferAllocations runs immediately after "
               "LowerOpaqueBlock and before host/device splitting";
      }
    }
  }

private:
  Map<Var, Map<String, Any>> metadata_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> consumed_;
};

} // namespace

tvm::transform::Pass ValidateTenstorrentKernelLaunch() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    TenstorrentLaunchValidator validator;
    validator.Validate(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.ValidateTenstorrentKernelLaunch", {});
}

tvm::transform::Pass LowerTenstorrentFrontendAnnotations() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return TenstorrentFrontendLowerer::Rewrite(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.LowerTenstorrentFrontendAnnotations", {});
}

tvm::transform::Pass LowerTenstorrentBufferAllocations() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    auto metadata =
        func->GetAttr<Map<Var, Map<String, Any>>>(kAllocBufferMetadata);
    if (!metadata) {
      return func;
    }
    TenstorrentBufferAllocationLowerer lowerer(metadata.value());
    func.CopyOnWrite()->body = lowerer(func->body);
    lowerer.VerifyConsumed();
    return WithoutAttr(std::move(func), kAllocBufferMetadata);
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.LowerTenstorrentBufferAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tenstorrent.transform.ValidateTenstorrentKernelLaunch",
           ValidateTenstorrentKernelLaunch)
      .def("tl.tenstorrent.transform.LowerTenstorrentFrontendAnnotations",
           LowerTenstorrentFrontendAnnotations)
      .def("tl.tenstorrent.transform.LowerTenstorrentBufferAllocations",
           LowerTenstorrentBufferAllocations);
}

} // namespace transform
} // namespace tenstorrent
} // namespace tl
} // namespace tvm
