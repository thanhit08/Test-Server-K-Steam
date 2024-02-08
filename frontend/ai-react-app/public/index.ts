import * as tf from '@tensorflow/tfjs-core';
import * as gpu from '@tensorflow/tfjs-backend-webgl';
import * as cpu from '@tensorflow/tfjs-backend-cpu';
import * as qna from '@tensorflow-models/qna';
let break_treeshaking_hack: any = tf;
break_treeshaking_hack = cpu;
break_treeshaking_hack = gpu;