from jax._src.callback import core, effects, dispatch, mlir, sharding_impls,\
                                pure_callback_batching_rule, batching, ad, xc, \
                                _check_shape_dtype, util, lax_map
import jax.tree_util as tree_util
import functools
from typing import Callable, Sequence, Any

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True

class IOEffect(effects.Effect):
  __str__ = lambda _: "IO"

class OrderedIOEffect(effects.Effect):
  __str__ = lambda _: "OrderedIO"

_IOEffect = IOEffect()
_OrderedIOEffect = OrderedIOEffect()
effects.lowerable_effects.add_type(IOEffect)
effects.lowerable_effects.add_type(OrderedIOEffect)
effects.control_flow_allowed_effects.add_type(IOEffect)
effects.control_flow_allowed_effects.add_type(OrderedIOEffect)
effects.ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(*args, result_avals, callback: Callable[..., Any],
                     ordered: bool):
  del result_avals, ordered
  return callback(*args)
io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))

@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(*avals, callback: Callable[..., Any],
                              result_avals, ordered: bool):
  del avals, callback
  effect = _OrderedIOEffect if ordered else _IOEffect
  return result_avals, {effect}

def io_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support JVP.")
ad.primitive_jvps[io_callback_p] = io_callback_jvp_rule

def io_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support transpose.")
ad.primitive_transposes[io_callback_p] = io_callback_transpose_rule

def io_callback_batching_rule(args, dims, *, callback, vectorized: bool,
                                result_avals: Sequence[core.ShapedArray],
                                ordered: bool):
  axis_size = next(a.shape[0] for a, d in zip(args, dims)
                   if d is not batching.not_mapped)
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  if vectorized:
    result_avals = tuple(
        core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)  # type: ignore
        for aval in result_avals)
    outvals = io_callback_p.bind(
        *new_args, callback=callback, vectorized=vectorized,
        result_avals=result_avals)
  else:
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)
    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return io_callback_p.bind(
          *merged_args, callback=callback, result_avals=result_avals,
          vectorized=vectorized)
    outvals = lax_map(_batch_fun, batched_args)
  return tuple(outvals), (0,) * len(outvals)
torized=False, result_avals=result_avals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule

def io_callback_lowering(ctx, *args, callback, ordered, **params):

  def _callback(*flat_args):
    return tuple(io_callback_impl(*flat_args, callback=callback,
                                  ordered=ordered, **params))

  # TODO(sharadmv): figure out the best API for sharding callbacks. For now, we
  # can only safely maximally shard. Should we allow device_index to be passed
  # in like host_callback?
  if isinstance(ctx.module_context.axis_context,
                (sharding_impls.SPMDAxisContext, sharding_impls.ShardingContext)):
    # Apply maximal sharding so pjit only executes the callback on device 0.
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
  else:
    sharding = None

  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)[0]
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
    ctx.set_tokens_out(mlir.TokenSet({_OrderedIOEffect: (token,)}))
  else:
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(io_callback_p, io_callback_lowering)

def io_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                *args: Any, ordered: bool = False, **kwargs: Any):
  """Calls an impure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. It is assumet to be an impure function.
      If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
      more efficient execution.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
    *args: arguments to be passed to the callback function
    ordered: boolean specifying whether sequential calls to callback must be ordered.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(callback(*args, **kwargs))

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  flat_args = map(core.raise_as_much_as_possible, flat_args)
  out_flat = io_callback_p.bind(
      *flat_args, callback=_flat_callback,
      result_avals=tuple(flat_result_avals),
      ordered=ordered)
  return tree_util.tree_unflatten(out_tree, out_flat)