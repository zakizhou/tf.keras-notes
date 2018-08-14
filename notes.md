```
def streaming_restore(status, session=None):
|__ status.run_restore_ops(session=session)
    // any restore operations are run by this `new_restore_ops_callback`
    status._checkpoint.new_restore_ops_callback = (lambda ops: session.run(ops, feed_dict=status._feed_dict))

class _CheckpointRestoreCoordinator
|__ __init__
    |__ self.restore_ops = []
        self.restore_ops_by_name = {}
        self.new_restore_ops_callback = None
        
|__ new_restore_ops(self, new_ops):
    self.restore_ops.extend(new_ops)
    if self.new_restore_ops_callback:
    |__ self.new_restore_ops_callback(new_ops)

class _CheckpointPosition
|__ restore
    |__ restore_ops = checkpointable._restore_from_checkpoint_position(self)
        self._checkpoint.new_restore_ops(restore_ops)

CheckpointableSaver(Object)
|__ restore
    |__ checkpoint = _CheckpointRestoreCoordinator(object_graph_proto=object_graph_proto,
                                                    save_path=file_prefix_tensor,
                                                    dtype_map=dtype_map)
        base._CheckpointPosition(checkpoint=checkpoint, proto_id=0).restore(self._root_checkpointable)
        load_status = CheckpointLoadStatus(checkpoint, root_checkpointable=self._root_checkpointable, feed_dict=file_prefix_feed_dict)

Checkpointable(base.CheckpointableBase)

Checkpoint(tracking.Checkpointable)
|__ __init__
    |__ self._saver = CheckpointableSaver(weakref.ref(self))

------------------------------------------------------------------------------------------------------------------------

CheckpointableDataStructure(base.CheckpointableBase)
|__ 

List(CheckpointableDataStructure, collections.Sequence)
|__

_ListWrapper(List, collections.MutableSequence, list)

Mapping(CheckpointableDataStructure, collections.Mapping)

_DictWrapper(Mapping, collections.MutableMapping)

------------------------------------------------------------------------------------------------------------------------

tensorflow/python/training/checkpointable/base.py
CheckpointableBase(Object)
|__ _maybe_initialize_checkpointable
    |__ self._unconditional_checkpoint_dependencies = []
        // used to handle deferred dependencies
        self._unconditional_deferred_dependencies = {}

|__ _add_variable_with_custom_getter
    // new variable is created with a custom `getter`
    |__ new_variable = getter(name=name, shape=shape, dtype=dtype, initializer=initializer, **kwargs_for_getter)
        // then track this newly created variable as checkpoint dependency
        self._track_checkpointable(new_variable, name=name, overwrite=overwrite)

|__ _track_checkpointable
    |__ current_object = self._lookup_dependency(name)
    // if denpendency if found, then if must be a layer, not variable, since variable has not be created, at this time
    // add the layer to the dependency of this `tf.keras.Model`
    |__ if current_object is not None
        // self._unconditional_checkpoint_dependencies is modified here
        |__ self._unconditional_checkpoint_dependencies[index] = new_reference
    // if it is None, then is must be a variable, handle deferred dependencies
    |__ elif current_object is None:
        |__ self._handle_deferred_dependencies(name=name, checkpointable=checkpointable)
    
|__ _handle_deferred_dependencies
    |__ deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    |__ for checkpoint_position in sorted(deferred_dependencies_list, 
                                          key=lambda restore: restore.checkpoint.restore_uid,
                                          reverse=True):
        // at this time, _CheckpointPosition.restore is called, then restored
        |__ checkpoint_position.restore(checkpointable)
        
|__ _restore_from_checkpoint_position
    |__ visit_queue = collections.deque([checkpoint_position])
        restore_ops = []
    |__ while visit_queue:
        current_position = visit_queue.popleft()
        restore_ops.extend(nest.flatten(current_position.checkpointable._single_restoration_from_checkpoint_position(
                                          checkpoint_position=current_position,
                                          visit_queue=visit_queue)))
                                          
|__ _single_restoration_from_checkpoint_position
    |__ for child in checkpoint_position.object_proto.children:
        |__ child_position = _CheckpointPosition(checkpoint=checkpoint, proto_id=child.node_id)
            local_object = self._lookup_dependency(child.local_name)
        |__ if local_object is None:
            |__ self._deferred_dependencies.setdefault(child.local_name, []).append(child_position)
        |__ else:
            |__ if child_position.bind_object(checkpointable=local_object):
                visit_queue.append(child_position)
                
------------------------------------------------------------------------------------------------------------------------

tensorflow/python/keras/engine/base_layer.py
Layer(CheckpointableBase)
|__ __init__
    // during constructor, empty lists for storing variables are created
    |__ self._trainable_weights = []
        self._non_trainable_weights = []

|__ add_weight
    // call _add_variable_with_custom_getter to create variable, note that this will add the created variable to the
    // checkpoint dependency of this layer
    |__ _add_variable_with_custom_getter
        |__ self._track_checkpointable(new_variable, name=name, overwrite=overwrite)
    // later append created variable to self._trainable_weights or self._non_trainable_weights according to whether it
    // trainable.
    // 
    |__ if trainable:
        |__ self._trainable_weights.append(variable)
        else:
        |__ self._non_trainable_weights.append(variable)

// used to collect all trainable and non trainable weights, typically added by self.add_weight
@property
|__ def weights(self):
    |__ return self.trainable_weights + self.non_trainable_weights

@property
|__ def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

@property
|__ def non_trainable_weights(self):
    |__ if self.trainable:
        |__ return self._non_trainable_weights
        else:
        |__ return self._trainable_weights + self._non_trainable_weights

|__ get_weights
|__ set_weights

// two methods that will be overridden for subclasses
// (1)
|__ build
    // variables should be created with self.add_weights in this method
    
|__ call
    // do computation logic here with inputs and created variables in self.build


|__ __call__
    // judge need to build graph or in in_deferred_mode, normally, build_graph should be True(if not in eager mode), 
    // in_deferred_mode should be False
    |__ build_graph = not context.executing_eagerly()
        in_deferred_mode = isinstance(input_list[0], DeferredTensor)
        
    // prepare input_shapes for building the layer
    |__ input_shapes = None
        if all(hasattr(x, 'shape') for x in input_list):
        |__ input_shapes = nest.map_structure(lambda x: x.shape, inputs)
        // self.built is False after constructor, build is completed here(variables are created), this check makes
        // sure that layer will always be built before self.call is called, so users can safely manipulate variables
        // created in self.build
        if not self.built:
        |__ self.build(input_shapes)
    
    // not in deferred mode(should never be in deferred mode)
    |__ if not in_deferred_mode:
        |__ outputs = self.call(inputs, *args, **kwargs)
    
    // Why need here? since self.build is called
    |__ self.built = True
    
    // note that, layers in tf.keras.layers don't have `_symbolic_set_inputs` or self.inputs, so the code block will not execute.
    // After constructor, then Model has '_symbolic_set_inputs' and not self.inputs is True, so self._symbolic_set_inputs
    // is call to build the Model
    |__ if hasattr(self, '_symbolic_set_inputs') and not self.inputs:
        // Subclassed network: explicitly set metadata normally set by a call to
        // self._set_inputs(). This is not relevant in eager execution.
        |__ self._symbolic_set_inputs(inputs, outputs)
    
    // this method is None for tf.keras.layers, but is not for Network
    |__ self._post_build_cleanup()


------------------------------------------------------------------------------------------------------------------------

tensorflow/python/keras/engine/network.py
Network(Layer)
// in __init__, if inputs and outputs are passed, then call `self._init_graph_network` to build graph network, otherwise
// call `self._init_subclassed_network` to build subclassed network, note that at this time, after __init__, model is
// not build, building happens at the first call of this model
|__ __init__
    // graph network
    |__ _init_graph_network
        |__ 
    
    // subclassed network
    |__ _init_subclassed_network(self, name=None):
        |__ self._base_init(name=name)
            |__ self.trainable = True
                self._is_compiled = False
                self.optimizer = None
                // for holding all layers
                self._layers = []
                
                // used for collecting variables that are directly assigned as attributes with constructor
                self._extra_variables = []
                
            // a saver is set, used for save/restore
            |__ self._checkpointable_saver = checkpointable_utils.CheckpointableSaver(weakref.ref(self))
            // Runs restore operations when graph building. (only applicable to subclassed Models)
            |__ self._in_progress_restore_finalizer = None
            self._is_graph_network = False
            // self.inputs, outputs is set to None, self.built is set to False since the constructore does not do building
            self.outputs = None
            self.inputs = None
            self.built = False
            
|__ __setattr__(name, value)
    // add checkpointable dependency here
    |__ value = data_structures.sticky_attribute_assignment(checkpointable=self, value=value, name=name)
            |__ checkpointable._track_checkpointable(value, name=name, overwrite=True)
        
    // if the set value is a subclass of Layer, Network, CheckpointableDataStructure, then track it, add it to self._layers
    |__ if isinstance(value, (base_layer.Layer,
                              Network,
                              data_structures.CheckpointableDataStructure))
        // trace variables here
        // in subclasses model, is_graph_network is False
        |__ if not is_graph_network:
            // avoid append duplicated layers
            if not any((layer is value for layer in self._layers)):
              self._layers.append(value)
              // always created resource variables for subclass model
              if hasattr(value, '_use_resource_variables'):
              |__ value._use_resource_variables = True
    
    // trace attached variables here
    |__ if (not no_dependency and isinstance(value, checkpointable.CheckpointableBase)):
        |__ if (  # For subclassed models only, users may add extra weights/variables # simply by assigning them to attributes.
                not self._is_graph_network
                and isinstance(value, variables.Variable)):
            // add variable to self._extra_variables
            |__ self._extra_variables.append(value)
    
    // do not forget this ~
    |__ super(Network, self).__setattr__(name, value)
    
        


// add_variable is forbidden for Network(and thus Model)
|__ add_variable
    |__ raise NotImplementedError

// overriden from tf.keras.layers.Layer, layers like tf.keras.layers.Dense will return self._trainable_weights but 
// a Network does not have it, returned is the variables belong to all layers in this Network
@property
|__ def trainable_weights(self):
    return checkpointable_layer_utils.gather_trainable_weights(trainable=self.trainable,
                                                               sub_layers=self.layers,
                                                               extra_variables=self._extra_variables)
    // recursive is used here, is layer is Network, call this method, if layer is tf.keras.layers.Layer, call tf.keras.layers.Layer.trainable_weights
    |__ for layer in sub_layers:
        |__ weights += layer.trainable_weights
    
    // extra variables are add here
    |__ trainable_extra_variables = [v for v in extra_variables if v.trainable]
        return weights + trainable_extra_variables

@property
|__ def non_trainable_weights(self):
    |__ checkpointable_layer_utils.gather_non_trainable_weights(trainable=self.trainable,
                                                                sub_layers=self.layers,
                                                                extra_variables=self._extra_variables)
        
// run queued restore, once called, weights will be retored from checkpoint
|__ _post_build_cleanup
    |__ if self._in_progress_restore_finalizer is not None:
        // Runs queued restore operations left over from load_weights when graph
        // building.
        |__ self._in_progress_restore_finalizer()
        |__ self._in_progress_restore_finalizer = None

// for get weights in memory
|__ get_weights

// for set weights from memory
|__ set_weights

// for save weights to disk
|__ save_weights
    // call self._checkpointable_saver.save
    |__ self._checkpointable_saver.save(filepath, session=session)

// for load weights from dish
|__ load_weights
    // get backen session
    |__ session = backend.get_session()
        // create a partial function with session fixed
        finalizer = functools.partial(status.run_restore_ops, session=session)
        // if this model has been built, means all variables have been set, then directly run the partial
        // function to retore variables
        |__ if self.built:
            |__ finalizer()
        // Hold on to this status object until the network is built (for
        // subclassed Models). Then we'll run restore ops if necessary.
        |__ else:
            // set self._in_progress_restore_finalizer, now it is not None
            |__ self._in_progress_restore_finalizer = finalizer
            
------------------------------------------------------------------------------------------------------------------------
tensorflow/python/keras/engine/training.py

tf.keras.Model

           
// Set model's input and output specs based on the input data received.
// This is to be used for Model subclasses, which do not know at instantiation time what their inputs look like.
|__ _set_inputs
    // call self._symbolic_set_inputs in graph mode
    |__ self._symbolic_set_inputs(inputs, training=training)
        |__ self.inputs = []
            self.input_names = []
        // inputs to list
        |__ if isinstance(inputs, (list, tuple)):
            |__ inputs = list(inputs)
            else:
            |__ inputs = [inputs]
        // for every input in inputs
        |__ for i, v in enumerate(inputs)
            // build a new name for this input
            |__ name = 'input_%d' % (i + 1)
                self.input_names.append(name)
            
            // if input is a Numpy
            |__ if isinstance(v, (np.ndarray)):
                // create a placeholder and add it to inputs
                |__ shape = (None,) + v.shape[1:]
                    placeholder = K.placeholder(shape=shape, name=name)
                    self.inputs.append(placeholder)
            |__ else:
                // assume here v is a tensor, directly add it to inputs
                |__ self.inputs.append(v)
        // if outputs is None
        |__ if outputs is None:
            |__ outputs = self.call(self.inputs[0])
        // reshape outputs to a list
        |__ if isinstance(outputs, (list, tuple)):
              outputs = list(outputs)
            else:
              outputs = [outputs]
        // set outputs, output_names, and set self.built to True
        // Note this is quite important since after self.built set to Truem self.compile can
        // True compile everything need to train
        |__ self.outputs = outputs
            self.output_names = ['output_%d' % (i + 1) for i in range(len(self.outputs))]
            self.built = True
            
            
|__ compile
    // if model is not built, only do these
    |__ self.optimizer = optimizers.get(optimizer) [TFOptimizer]
        self.loss = loss
        self.metrics = metrics or []
        // maybe useful for model with multi outputs such as Mask RCNN
        self.loss_weights = loss_weights
    
    ------
    // if self.built: now can truly compile this model
    |__ self._is_compiled = True
    
    // in graph mode
    // set loss functions[get loss functions according to self.loss] and also weighted loss functions
    |__ self.loss_functions = loss_functions
        weighted_losses = [training_utils.weighted_masked_objective(fn) for fn in loss_functions]
    
    // Prepare targets of model.
    |__ self.targets = []
        // for each output, add a target to self.targets
        |__ for i in range(len(self.outputs)):
            self.targets.append(target)
    
    // Prepare total_loss of model
    |__ total_loss = None
        |__ for i in range(len(self.outputs)):
            // for each output, get output and target
            |__ y_true = self.targets[i]
                y_pred = self.outputs[i]
            // get the (weighted) loss function
            |__ weighted_loss = weighted_losses[i]
                sample_weight = sample_weights[i]
                mask = masks[i]
                loss_weight = loss_weights_list[i]
            // call loss function with output and target
            |__ output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
                // add weighted loss to total_loss
                total_loss += loss_weight * output_loss
            
            // set total_loss tensor of model
            self.total_loss = total_loss
    
    
    // Functions for train, test and predict will
    // be compiled lazily when required.
    // This saves time when the user is not using all functions.
    |__ self._function_kwargs = kwargs
    
    // self.train_function is quite important for actual training, see _make_train_function
    |__ self.train_function = None
        self.test_function = None
        self.predict_function = None

// backpropagation is done in this function
|__ _make_train_function
    `updates` should contain [op for adding iterations, apply_gradients of optimzer]
    |__ updates = self.optimizer.get_updates(params=self._collected_trainable_weights, loss=self.total_loss)
        |__ self.updates = [state_ops.assign_add(self.iterations, 1)]
            grads = self.optimizer.compute_gradients(loss, params)
            // `opt_update` is actually the optimization step 
            opt_update = self.optimizer.apply_gradients(grads, global_step=self.iterations)
            // now self.updates contains the op for optimization
            self.updates.append(opt_update)
            
    // self.train_function is set
    |__ self.train_function = K.function(inputs, [self.total_loss] + self.metrics_tensors,
                                         updates=updates,
                                         name='train_function',
                                         **self._function_kwargs)
        // in the constructor of Function
        |__ Function
            |__ self.updates_op = control_flow_ops.group(*updates_ops)
            
            |__ def _make_callable
                |__ callable_opts.fetch.append(x.name)
                    callable_opts.target.append(self.updates_op.name)
                    // `session._make_callable_from_options` to build a callable_fn
                    callable_fn = session._make_callable_from_options(callable_opts)
                    self._callable_fn = callable_fn
            
            // self.train_function.__call__ is overriden, in each call if train_function, updates_op will be run
            |__ __call__
                // first need a session, get it from `tf.keras.backend.get_session()`
                |__ session = get_session()
                |__ fetched = self._callable_fn(*array_vals)
    
    // run queued restore, at this time, weights are retored from checkpoint, note that in graph mode, self.train_function
    // hasn't been called before weights are restored if `load_weights` so once self.train_function is called, all the weights
    // are now the ones retored in checkpoint
    |__ self._post_build_cleanup()
            
            
|__ fit
    # Validate and standardize user data.
    # Runs validation checks on input and target data passed by the user.

    # Also standardizes the data to lists of arrays, in order.

    # Also builds and compiles the model on the fly if it is a subclassed model
    # that has never been called before (and thus has no inputs/outputs).
    |__ x, y, sample_weights = self._standardize_user_data(
                                    x,
                                    y,
                                    sample_weight=sample_weight,
                                    class_weight=class_weight,
                                    batch_size=batch_size,
                                    check_steps=True,
                                    steps_name='steps_per_epoch',
                                    steps=steps_per_epoch,
                                    validation_split=validation_split)
        # when x is a Dataset, y should be None
        |__ x = x.make_one_shot_iterator()  # in eager mode
            iterator = x.make_initializable_iterator()  # in graph mode
        
        |__ x, y = next_element
        
        // First, we build/compile the model on the fly if necessary.
        |__ all_inputs = []
            is_build_called = False
            is_compile_called = False
            
        |__ if not self.built:
            // Build the model using the retrieved inputs (value or symbolic).
            // If values, then in symbolic-mode placeholders will be created
            // to match the value shapes.
            |__ if not self.inputs:
                is_build_called = True
                // self.inputs, input_names, outputs, output_names is set, self.built is set to True
                self._set_inputs(x)
        
        |__ if y is not None:
            |__ if not self._is_compiled:
                // another compile, at this time, model is built, so 
                // self.loss_functions, self.targets, self.total_loss is set
                |__ self.compile
    
    // in eager mode, training_eager.py
    |__ fit_loop
        // loop for epoches
        |__ for epoch in range(initial_epoch, epochs):
            |__ if steps_per_epoch is not None:
                // when inputs are from tf.data.Dataset/Iterator, `steps_per_epoch` is not None, at this time
                // call `iterator_fit_loop`
                |__ iterator_fit_loop
                    |__ _process_single_batch
                        // put foward computation under GradientTape to record all computations during forward pass
                        |__ with GradientTape() as tape:
                                outs, loss, loss_metrics = _model_loss(model, inputs, targets,
                                                                       sample_weights=sample_weights,
                                                                       training=training)
                        // call GradientTape.gradient() to calcualte gradients w.r.t all trainable variables
                        |__ grads = tape.gradient(loss, model._collected_trainable_weights)
                            // apply gradients to variables to do updating
                            model.optimizer.apply_gradients(zip(grads, model._collected_trainable_weights))
            
            |__ else:
                // at this time, inputs are from numpy, so called batch_fit_loop, at this time, batch_size argument
                // is valid
                |__ batch_fit_loop
                    |__ for batch_index, (batch_start, batch_end) in enumerate(batches):
    
    // in graph mode, training_arrays.py
    |__ fit_loop
        // model.train_function is set here for training, also restore weights if users do `Model.load_weights`
        |__ model._make_train_function()
            f = model.train_function
            
        // loop for epoches
        |__ for epoch in range(initial_epoch, epochs):
            |__ if steps_per_epoch is not None:
                |__ for step_index in range(steps_per_epoch):
                    // try to do a `Session.run` on a batch data
                    |__ try:
                        |__ outs = f(ins)
                    // if OutOfRangeError, tell users dataset is out of data
                    |__ except errors.OutOfRangeError:
------------------------------------------------------------------------------------------------------------------------  
tensorflow/python/keras/engine/training_arrays.py
fit_loop
// call Model._make_train_function() to build Model.train_function
|__ model._make_train_function()
    f = model.train_function
            
RNN(Layer)
|__ call
    // define a step function
    |__ def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)
    |__ last_output, outputs, states = K.rnn(step,
                                            inputs,
                                            initial_state,
                                            constants=constants,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            unroll=self.unroll,
                                            input_length=timesteps)
```

behaviors in different stages of tf.keras.Model

```
|__ constructor:
    Model.layers                                list<tf.keras.layers.Layer>
    
    Model._in_progress_restore_finalizer        None
        
    Model.inputs                                None
    Model.outputs                               None
    Model.built                                 False
    
    Model.variables                             []
    Model.weights                               []
    
    Model.optimizer                             None
    Model.loss                                  Error
    Model.metrics                               Error
    Model.loss_weights                          Error
    
    Model.losses                                []
    Model.updates                               []
    Model._is_compiled                          False
    


// possible
|__ load_weights
    // not built this time
    Model.layers                                list<tf.keras.layers.Layer>
    
    Model._in_progress_restore_finalizer        callable
        
    Model.inputs                                None
    Model.outputs                               None
    Model.built                                 False
    
    Model.variables                             []
    Model.weights                               []
    
    
    Model.optimizer                             None
    Model.loss                                  Error
    Model.metrics                               Error
    Model.loss_weights                          Error
    
    Model.losses                                []
    Model.updates                               []
    Model._is_compiled                          False
    

// still not built, only four things set
|__ compile
    // not built this time, so true compile happens at fit method
    Model.layers                                list<tf.keras.layers.Layer>
    
    Model._in_progress_restore_finalizer        callable
        
    Model.inputs                                None
    Model.outputs                               None
    Model.built                                 False
    
    Model.variables                             []
    Model.weights                               []
    
    
    Model.optimizer                             tf.keras.optimizers.Optimizer
    Model.loss                                  list<str> | list<callable>
    Model.metrics                               list<str>
    Model.loss_weights                          None
    
    Model.losses                                []
    Model.updates                               []
    Model._is_compiled                          False
    
    
        
|__ fit
    |__ _standardize_user_data
        |__ x, y ---> Numpy
            x, y ---> Dataset/Tensor
        
        if not self.built:
        |__ self._set_inputs(x)
            Model.layers                                list<tf.keras.layers.Layer>
            
            Model._in_progress_restore_finalizer        callable
                
            Model.inputs                                list<tf.Tensor>
            Model.input_names                           list<str>
            Model.outputs                               list<tf.Tensor>
            Model.output_names                          list<str>
            Model.built                                 True
            
            Model.variables                             list<tf.ResourceVariable>
            Model.weights                               list<tf.ResourceVariable>
            
            Model.optimizer                             tf.keras.optimizers.Optimizer
            Model.loss                                  list<str> | list<callable>
            Model.metrics                               list<str>
            Model.loss_weights                          None
            
            Model.loss_functions                        callable
            Model.target_tensors                        None
            Model.targets                               Error
            Model.losses                                []
            Model._is_compiled                          False
            
            Model.updates                               []


            
        if y is not None:
        |__ self.compile(optimizer=self.optimizer,
                         loss=self.loss,
                         metrics=self.metrics,
                         loss_weights=self.loss_weights,
                         target_tensors=target_tensors)

            Model.layers                                list<tf.keras.layers.Layer>
            
            Model._in_progress_restore_finalizer        callable
                
            Model.inputs                                list<tf.Tensor>
            Model.input_names                           list<str>
            Model.outputs                               list<tf.Tensor>
            Model.output_names                          list<str>
            Model.built                                 True
            
            Model.variables                             list<tf.ResourceVariable>
            Model.weights                               list<tf.ResourceVariable>
            
            Model.optimizer                             tf.keras.optimizers.Optimizer
            Model.optimizer.updates                     list<tf.Operation>
            Model.loss                                  list<str> | list<callable>
            Model.metrics                               list<str>
            Model.loss_weights                          None
            
            Model.loss_functions                        callable
            Model.target_tensors                        None | list<tf.Tensor>
            Model.targets                               list<tf.Tensor>
            Model.losses                                []
            Model._is_compiled                          True
            
            Model.updates                               []
            
            Model.train_function                        tf.keras.backend.Function
            
        
    |__ fit_loop
        |__ _make_train_function
            |__ _post_build_cleanup
            
            Model.layers                                list<tf.keras.layers.Layer>
            
            Model._in_progress_restore_finalizer        callable
                
            Model.inputs                                list<tf.Tensor>
            Model.input_names                           list<str>
            Model.outputs                               list<tf.Tensor>
            Model.output_names                          list<str>
            Model.built                                 True
            
            Model.variables                             list<tf.ResourceVariable>
            Model.weights                               list<tf.ResourceVariable>
            
            Model.optimizer                             tf.keras.optimizers.Optimizer
            Model.optimizer.updates                     list<tf.Operation>
            Model.loss                                  list<str> | list<callable>
            Model.metrics                               list<str>
            Model.loss_weights                          None
            
            Model.loss_functions                        callable
            Model.target_tensors                        None | list<tf.Tensor>
            Model.targets                               list<tf.Tensor>
            Model.losses                                []
            Model._is_compiled                          True
            
            Model.updates                               []
            
            Model.train_function                        tensorflow.python.keras.backend.Function
        
        |__ 

```




Built-in layers/modules



| layers/modules        | `tf.keras.layers`          | `torch.nn`  |
| ------------- |:-------------:| -----:|
| embedding      | `Embedding` | `Embedding` |
| dense/fully-conncted      | `Dense`      |   `Linear` |
| 2d convolution  | `Conv2D`      |    `Conv2d` |
| 2d convolution transpose| `Conv2DTranspose` | `ConvTranspose2d`|
|LSTM|`LSTM`(CPU), `CuDNNLSTM`(GPU)|`LSTM`(CPU & GPU)|
|GRU|`GRU`(CPU), `CuDNNGRU`(GPU)|`GRU`(CPU & GPU)|

Container

| Functionality          | `tf.keras.Model`         | `torch.nn.Module` |
| ------------- |:-------------:| -----------------------------------:|
|forward computation| `Model.call`|`Module.forward`|
| variables/parameters      | `Model.variables` | `Module.parameters()`   |
| sub-modules/models      | `Model.layers`      |   `Module.modules`   |
| save   |    `Model.save_weights`   | `Module.state_dict()` |
| load  |    `Model.load_weights`   | `Module.load_state_dict()` |


behaviors in different stages of tf.keras.Model
```Python
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        # define your submodels/layers here
        self.d = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=True, mask=None):
        # define your computational logic here
        return  self.d(inputs)


# (1) __init__ : constructor
m = Model(...)
# (2) load_weights : restore  ->  possibly restore weights
m.load_weights("path/to/store/checkpoint")
# (3) compile : compile I
m.compile(optimizer=tf.keras.optimizers.Adam(),
          loss=["whatever_loss_name"])
# (4) fit/_standardize_user_data/_set_inputs : build  ->  build the model, defines computation for predictions
m._set_inputs(tf.placeholder(tf.float32, [None, 784]))
# (5) compile : compile Ⅱ
m.compile(optimizer=m.optimizer,
          loss=m.loss)
# (6) train :
m._make_train_function()
```

## `__init__`
`m.layers`


Tables

|              attributes              |          `__init__`         |        `load_weights`       |           `compile`           | `fit/_standardize_user_data/_set_inputs` | `fit/_standardize_user_data/compile` | `fit/fit_loop/_make_train_function` |
|:------------------------------------:|:---------------------------:|:---------------------------:|:-----------------------------:|:----------------------------------------:|:------------------------------------:|:-----------------------------------:|
| Model.layers                         | list<tf.keras.layers.Layer> | list<tf.keras.layers.Layer> |  list<tf.keras.layers.Layer>  |        list<tf.keras.layers.Layer>       |      list<tf.keras.layers.Layer>     |     list<tf.keras.layers.Layer>     |
| ---------stage: Restore ------------ | --------------------------- | --------------------------- |  ---------------------------  |  --------------------------------------  |   ---------------------------------  |   --------------------------------  |
| Model._in_progress_restore_finalizer |             None            |           callable          |            callable           |                 callable                 |               callable               |                 None                |
| ---------stage: Compile I ---------- | --------------------------- | --------------------------- |  ---------------------------  |   -------------------------------------  |   ---------------------------------  |   --------------------------------  |
| Model.optimizer                      |             None            |             None            | tf.keras.optimizers.Optimizer |       tf.keras.optimizers.Optimizer      |     tf.keras.optimizers.Optimizer    |    tf.keras.optimizers.Optimizer    |
| Model.loss                           |              /              |              /              |   list<str> / list<callable>  |        list<str> / list<callable>        |      list<str> / list<callable>      |      list<str> / list<callable>     |
| Model.metrics                        |              /              |              /              |           list<str>           |                 list<str>                |               list<str>              |              list<str>              |
| Model.loss_weights                   |              /              |              /              |              None             |                   None                   |                 None                 |                 None                |
| --------- stage: build ------------- | --------------------------- | --------------------------- |  ---------------------------  |   -------------------------------------  |   ---------------------------------  |   --------------------------------  |
| Model.built                          |            False            |            False            |             False             |                   True                   |                 True                 |                 True                |
| Model.inputs                         |              []             |              []             |               []              |              list<tf.Tensor>             |            list<tf.Tensor>           |           list<tf.Tensor>           |
| Model.input_names                    |              /              |              /              |               /               |                 list<str>                |               list<str>              |              list<str>              |
| Model.outputs                        |              []             |              []             |               []              |              list<tf.Tensor>             |            list<tf.Tensor>           |           list<tf.Tensor>           |
| Model.output_names                   |              /              |              /              |               /               |                 list<str>                |               list<str>              |              list<str>              |
| Model.variables                      |              []             |              []             |               []              |         list<tf.ResourceVariable>        |       list<tf.ResourceVariable>      |      list<tf.ResourceVariable>      |
| Model.weights                        |              []             |              []             |               []              |         list<tf.ResourceVariable>        |       list<tf.ResourceVariable>      |      list<tf.ResourceVariable>      |
| --------- stage: Compile Ⅱ --------- | --------------------------- | --------------------------- |  ---------------------------  |   -------------------------------------  |   ---------------------------------  |   --------------------------------  |
| Model._is_compiled                   |            False            |            False            |             False             |                   False                  |                 True                 |                 True                |
| Model.loss_functions                 |              /              |              /              |               /               |                     /                    |            list<callable>            |            list<callable>           |
| Model.target_tensors                 |              /              |              /              |               /               |                     /                    |        None / list<tf.Tensor>        |        None / list<tf.Tensor>       |
| Model.targets                        |              /              |              /              |               /               |                     /                    |            list<tf.Tensor>           |           list<tf.Tensor>           |
| Model.losses                         |              []             |              []             |               []              |                    []                    |                  []                  |                  []                 |
| -                                    |              -              |              -              |               -               |                     -                    |                   -                  |                  -                  |
| Model.updates                        |              []             |              []             |               []              |                    []                    |                  []                  |                  []                 |
| --------- stage: Train ------------- | --------------------------- | --------------------------- |  ---------------------------  |   -------------------------------------  |   ---------------------------------  |   --------------------------------  |
| Model.optimizer.updates              |              /              |              /              |               /               |                     /                    |                   /                  |          list<tf.Operation>         |
| Model.train_function                 |              /              |              /              |               /               |                     /                    |                   /                  |      tf.keras.backend.Function      |
  
```
╔══════════════════════════════════════╦═════════════════════════════╦═════════════════════════════╦═══════════════════════════════╦══════════════════════════════════════════╦══════════════════════════════════════╦═════════════════════════════════════╗
║              attributes              ║          `__init__`         ║        `load_weights`       ║           `compile`           ║ `fit/_standardize_user_data/_set_inputs` ║ `fit/_standardize_user_data/compile` ║ `fit/fit_loop/_make_train_function` ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.layers                         ║ list<tf.keras.layers.Layer> ║ list<tf.keras.layers.Layer> ║  list<tf.keras.layers.Layer>  ║        list<tf.keras.layers.Layer>       ║      list<tf.keras.layers.Layer>     ║     list<tf.keras.layers.Layer>     ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ ----------stage: Restore------------ ║ --------------------------- ║ --------------------------- ║  ---------------------------  ║  --------------------------------------  ║   ---------------------------------  ║   --------------------------------  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model._in_progress_restore_finalizer ║             None            ║           callable          ║            callable           ║                 callable                 ║               callable               ║                 None                ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ ---------stage: Compile I ---------- ║ --------------------------- ║ --------------------------- ║  ---------------------------  ║   -------------------------------------  ║   ---------------------------------  ║   --------------------------------  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.optimizer                      ║             None            ║             None            ║ tf.keras.optimizers.Optimizer ║       tf.keras.optimizers.Optimizer      ║     tf.keras.optimizers.Optimizer    ║    tf.keras.optimizers.Optimizer    ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.loss                           ║              /              ║              /              ║   list<str> / list<callable>  ║        list<str> / list<callable>        ║      list<str> / list<callable>      ║      list<str> / list<callable>     ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.metrics                        ║              /              ║              /              ║           list<str>           ║                 list<str>                ║               list<str>              ║              list<str>              ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.loss_weights                   ║              /              ║              /              ║              None             ║                   None                   ║                 None                 ║                 None                ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ ----------stage: build ------------  ║ --------------------------- ║ --------------------------- ║  ---------------------------  ║   -------------------------------------  ║   ---------------------------------  ║   --------------------------------  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.built                          ║            False            ║            False            ║             False             ║                   True                   ║                 True                 ║                 True                ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.inputs                         ║              []             ║              []             ║               []              ║              list<tf.Tensor>             ║            list<tf.Tensor>           ║           list<tf.Tensor>           ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.input_names                    ║              /              ║              /              ║               /               ║                 list<str>                ║               list<str>              ║              list<str>              ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.outputs                        ║              []             ║              []             ║               []              ║              list<tf.Tensor>             ║            list<tf.Tensor>           ║           list<tf.Tensor>           ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.output_names                   ║              /              ║              /              ║               /               ║                 list<str>                ║               list<str>              ║              list<str>              ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.variables                      ║              []             ║              []             ║               []              ║         list<tf.ResourceVariable>        ║       list<tf.ResourceVariable>      ║      list<tf.ResourceVariable>      ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.weights                        ║              []             ║              []             ║               []              ║         list<tf.ResourceVariable>        ║       list<tf.ResourceVariable>      ║      list<tf.ResourceVariable>      ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ ---------stage: Compile   ---------  ║ --------------------------- ║ --------------------------- ║  ---------------------------  ║   -------------------------------------  ║   ---------------------------------  ║   --------------------------------  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model._is_compiled                   ║            False            ║            False            ║             False             ║                   False                  ║                 True                 ║                 True                ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.loss_functions                 ║              /              ║              /              ║               /               ║                     /                    ║            list<callable>            ║            list<callable>           ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.target_tensors                 ║              /              ║              /              ║               /               ║                     /                    ║        None / list<tf.Tensor>        ║        None / list<tf.Tensor>       ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.targets                        ║              /              ║              /              ║               /               ║                     /                    ║            list<tf.Tensor>           ║           list<tf.Tensor>           ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.losses                         ║              []             ║              []             ║               []              ║                    []                    ║                  []                  ║                  []                 ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ -                                    ║              -              ║              -              ║               -               ║                     -                    ║                   -                  ║                  -                  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.updates                        ║              []             ║              []             ║               []              ║                    []                    ║                  []                  ║                  []                 ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ ----------stage: Train ------------- ║ --------------------------- ║ --------------------------- ║  ---------------------------  ║   -------------------------------------  ║   ---------------------------------  ║   --------------------------------  ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.optimizer.updates              ║              /              ║              /              ║               /               ║                     /                    ║                   /                  ║          list<tf.Operation>         ║
╠══════════════════════════════════════╬═════════════════════════════╬═════════════════════════════╬═══════════════════════════════╬══════════════════════════════════════════╬══════════════════════════════════════╬═════════════════════════════════════╣
║ Model.train_function                 ║              /              ║              /              ║               /               ║                     /                    ║                   /                  ║      tf.keras.backend.Function      ║
╚══════════════════════════════════════╩═════════════════════════════╩═════════════════════════════╩═══════════════════════════════╩══════════════════════════════════════════╩══════════════════════════════════════╩═════════════════════════════════════╝\
```                                  
                                                                   