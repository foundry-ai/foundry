Training Models
===============

foundry can automate a simple JAX training loop for you.

For instance, here is a short block of code
which trains a model on a dataset using the adam optimizer:

.. code-block:: python

    import foundry.train as tu

    def loss_fn(vars: Any, iteration: jax.Array,
                rng_key: jax.Array, batch: jax.Array) -> tu.LossOutput:
        loss = ...
        return tu.LossOutput(loss=loss, stats={"loss": loss})
    
    tu.fit(
        data=train_data,
        batch_loss_fn=tu.batch_loss(loss_fn),
        init_vars=init_vars,
        rng_key=jax.PRNGKey(42),
        optimizer=optax.adam(1e-3),
        max_epochs=10, # can specify either max_epochs or max_iterations
        hooks=[
            tu.every_epoch(tu.console_logger("train."))
        ]
    )
