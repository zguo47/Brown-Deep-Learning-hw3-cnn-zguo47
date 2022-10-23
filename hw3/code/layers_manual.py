import math
from unittest import result

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        if (h_in % sh == 0):
            pad_along_height = max(fh - sh, 0)
        else:
            pad_along_height = max(fh - (h_in % sh), 0)
        if (w_in % sw == 0):
            pad_along_width = max(fw - sw, 0)
        else:
            pad_along_width = max(fw - (w_in % sw), 0)

        # Cleaning padding input.
        # if self.padding == "SAME":
        #     ph = (fh - 1) // 2
        #     pw = (fw - 1) // 2
        # elif self.padding == "VALID":
        #     ph, pw = 0, 0
        # else:
        #     raise AssertionError(f"Illegal padding type {self.padding}")

        if self.padding == "SAME":
            ph_top = pad_along_height // 2
            ph_bottom = pad_along_height - ph_top
            pw_left = pad_along_width // 2
            pw_right = pad_along_width - pw_left
        elif self.padding == "VALID":
            ph_top = 0
            ph_bottom = 0
            pw_left = 0
            pw_right = 0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        ## TODO: Convolve filter from above with the inputs.
        ## Note: Depending on whether you used SAME or VALID padding,
        ## the input and output sizes may not be the same

        ## Pad input if necessary

        ## Calculate correct output dimensions

        ## Iterate and apply convolution operator to each image

        ## PLEASE RETURN A TENSOR using tf.convert_to_tensor(your_array, dtype=tf.float32)

        if self.padding == "VALID":
            output_height = math.ceil((h_in - fh + 1)/sh)
            output_width = math.ceil((w_in - fw + 1)/sw)
            outputs_shape = (bn, output_height, output_width, c_out)       

        if self.padding == "SAME":
            output_height = math.ceil(h_in/sh)
            output_width = math.ceil(w_in/sw)
            outputs_shape = (bn, int(output_height), int(output_width), c_out)
            paddings = tf.constant([[0, 0], [ph_top, ph_bottom], [pw_left, pw_right], [0, 0]])
            inputs = tf.pad(inputs, paddings, "CONSTANT")
        
        h_in += (ph_top+ph_bottom)
        w_in += (pw_left+pw_right)

        outputs = np.zeros(outputs_shape)
        for b in range(bn):
            for k in range(c_out):
                for h in range(outputs_shape[1]):
                    for w in range(outputs_shape[2]):
                        result = 0
                        for c in range(c_in):
                            kernel_flat = tf.reshape(self.kernel[:, :, c, k],self.kernel_size)
                            inputs_flat = tf.reshape(inputs[b, :, :, c],(h_in, w_in))
                            inputs_sliced = tf.slice(inputs_flat, [h*sh, w*sw], [fh, fw])
                            result += tf.reduce_sum(inputs_sliced*kernel_flat)
                        outputs[b, h, w, k] = result
                    
        outputs = tf.convert_to_tensor(outputs + self.bias, dtype=tf.float32)
        outputs = self.activation(outputs)
        
        return outputs
