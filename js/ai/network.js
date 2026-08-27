// Hand-rolled forward pass for the trained network - a faithful JS port of
// python/colourwars/network.py's ColourWarsNet, reading weights exported by
// python -m colourwars.export_weights (see js/ai/weights.js). No external ML
// library/runtime - the network is tiny (~480K params, fixed 7x7 input), so
// plain typed-array loops are fast enough without WASM.
//
// Every op here (3x3/1x1 conv with 'same' padding, eval-mode batchnorm,
// relu, residual add, linear, tanh) must match PyTorch's math exactly -
// see python/colourwars/tests/cross_check_js_network.py for the numerical
// verification against the original best.pt.
(function (root) {
  'use strict';

  function conv2dSame(input, cIn, h, w, weight, cOut, k) {
    // weight: nested [cOut][cIn][k][k], stride 1, no bias, 'same' padding
    // (pad = floor(k/2)) - matches nn.Conv2d(..., padding=1) for k=3 and
    // nn.Conv2d(..., kernel_size=1) (no padding needed) for k=1.
    var pad = Math.floor(k / 2);
    var out = new Float32Array(cOut * h * w);
    for (var oc = 0; oc < cOut; oc++) {
      var wOc = weight[oc];
      for (var r = 0; r < h; r++) {
        for (var c = 0; c < w; c++) {
          var sum = 0;
          for (var ic = 0; ic < cIn; ic++) {
            var wIc = wOc[ic];
            var inBase = ic * h * w;
            for (var kh = 0; kh < k; kh++) {
              var ir = r + kh - pad;
              if (ir < 0 || ir >= h) continue;
              var wKh = wIc[kh];
              var rowBase = inBase + ir * w;
              for (var kw = 0; kw < k; kw++) {
                var ic2 = c + kw - pad;
                if (ic2 < 0 || ic2 >= w) continue;
                sum += input[rowBase + ic2] * wKh[kw];
              }
            }
          }
          out[oc * h * w + r * w + c] = sum;
        }
      }
    }
    return out;
  }

  // In-place eval-mode batchnorm: y = (x - running_mean) / sqrt(running_var
  // + eps) * weight + bias, folded into a single scale+shift per channel -
  // numerically identical to PyTorch's BatchNorm2d.eval() formula.
  function batchNormInplace(x, c, h, w, bn) {
    var stride = h * w;
    for (var ch = 0; ch < c; ch++) {
      var scale = bn.weight[ch] / Math.sqrt(bn.var[ch] + bn.eps);
      var shift = bn.bias[ch] - bn.mean[ch] * scale;
      var base = ch * stride;
      for (var i = 0; i < stride; i++) {
        x[base + i] = x[base + i] * scale + shift;
      }
    }
  }

  function reluInplace(x) {
    for (var i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  }

  function addInplace(a, b) {
    for (var i = 0; i < a.length; i++) a[i] += b[i];
  }

  function linear(input, weight, bias) {
    // weight: [outDim][inDim] (matches nn.Linear.weight's PyTorch layout).
    var outDim = weight.length;
    var out = new Float32Array(outDim);
    for (var o = 0; o < outDim; o++) {
      var wRow = weight[o];
      var sum = bias ? bias[o] : 0;
      for (var i = 0; i < input.length; i++) sum += input[i] * wRow[i];
      out[o] = sum;
    }
    return out;
  }

  // stateFlat: Float32Array(numPlanes*rows*cols) from Encode.encodeState.
  // weights: the object exported to js/ai/weights.js (window.AI_WEIGHTS).
  // Returns { policyLogits: Float32Array(rows*cols), value: Float32Array(maxPlayers) }.
  function forward(stateFlat, weights) {
    var c = weights.channels, h = weights.rows, w = weights.cols;

    var x = conv2dSame(stateFlat, weights.numPlanes, h, w, weights.stem.convW, c, 3);
    batchNormInplace(x, c, h, w, weights.stem.bn);
    reluInplace(x);

    for (var b = 0; b < weights.resBlocks.length; b++) {
      var block = weights.resBlocks[b];
      var residual = x;
      var out1 = conv2dSame(x, c, h, w, block.conv1W, c, 3);
      batchNormInplace(out1, c, h, w, block.bn1);
      reluInplace(out1);
      var out2 = conv2dSame(out1, c, h, w, block.conv2W, c, 3);
      batchNormInplace(out2, c, h, w, block.bn2);
      addInplace(out2, residual);
      reluInplace(out2);
      x = out2;
    }

    var pConv = conv2dSame(x, c, h, w, weights.policy.convW, 2, 1);
    batchNormInplace(pConv, 2, h, w, weights.policy.bn);
    reluInplace(pConv);
    var policyLogits = linear(pConv, weights.policy.fcW, weights.policy.fcB);

    var vConv = conv2dSame(x, c, h, w, weights.value.convW, 4, 1);
    batchNormInplace(vConv, 4, h, w, weights.value.bn);
    reluInplace(vConv);
    var v1 = linear(vConv, weights.value.fc1W, weights.value.fc1B);
    reluInplace(v1);
    var v2 = linear(v1, weights.value.fc2W, weights.value.fc2B);
    for (var i = 0; i < v2.length; i++) v2[i] = Math.tanh(v2[i]);

    return { policyLogits: policyLogits, value: v2 };
  }

  root.NeuralNet = { forward: forward };
})(typeof window !== 'undefined' ? window : globalThis);
