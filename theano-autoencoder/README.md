# Theano auto-encoder
1st try to implement an encoder-decoder with Theano

Auto-encoder spec:
 - of the form: x-> f1(W1 x + b1) = h -> f2(W2 h + b2) = x'
 - tied weights (W2 = W1^T)
 - sigmoid activation function f1
 - sigmoid output function f2

python prereq:
 - numpy
 - theano

tutorial followed from : https://triangleinequality.wordpress.com/2014/08/12/theano-autoencoders-and-mnist/

related post : http://deeplearning.net/tutorial/dA.html
