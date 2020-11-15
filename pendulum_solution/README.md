## Pendulum Solution

### Solution hyper-parameters

```
buffer_size=50000
buffer_batch_size=64
exploration_value=0.01
discount_factor=0.99
tau=0.05
actor_learning_rate=0.001
critic_learning_rate=0.002
```

___


### Actor Summary

The out put layer of the actor had a uniform kernel_initializer with `minval=-0.003, maxval=0.003`. The activation functions on the two dense layers where `relu` and the final layer `tanh` to give out put in `[-1, 1]`.


```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 3)]               0         
_________________________________________________________________
dense (Dense)                (None, 256)               1024      
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
_________________________________________________________________
tf_op_layer_Mul (TensorFlowO [(None, 1)]               0         
=================================================================
Total params: 67,073
Trainable params: 67,073
Non-trainable params: 0
```


___



### Critic Summary

The activation functions on `dense_3`, `dense_4`, `dense_5` where `relu`. The output activation is `linear`. 

```
_________________________________________________________________
Model: "functional_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 4)            0           input_2[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 400)          2000        concatenate[0][0]                
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 400)          160400      dense_3[0][0]                    
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            401         dense_4[0][0]                    
==================================================================================================
Total params: 162,801
Trainable params: 162,801
Non-trainable params: 0
```
