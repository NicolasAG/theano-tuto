import theano
import theano.tensor as T
from theano import In

### Logistic Function:

x = T.fmatrix('x')
s = 1 / (1+T.exp(-x))
logistic = theano.function([x], [s])

print "logistic([[0,1],[-1,-2]]) ="
print logistic([[0, 1], [-1, -2]])
print "- - -"

### Computing More than one Thing at the Same Time

a, b = T.fmatrices('a', 'b')
diff = a-b
abs_diff = abs(diff)
diff_sqrd = diff**2
super_function = theano.function([a, b], [diff, abs_diff, diff_sqrd])

print "a = b = [[0,1],[-1,-2]]. [a-b, abs(a-b), (a-b)^2] ="
print super_function([[0, 1], [-1, -2]], [[0, 1], [-1, -2]])
print "- - -"

### Setting a Default Value for an Argument with theano.In()

x, y, w = T.dscalars('x', 'y', 'w')
z = (x+y)*w
f = theano.function([x, In(y, name='y', value=1.), In(w, name='w', value=1.)], [z])

print "f(2.3) = (2.3+1)*1 =", f(2.3)
print "f(2.3,2) = (2.3+2)*1 =", f(2.3, 2.)
print "f(2.3,2,2) = (2.3+2)*2 =", f(2.3, 2., 2.)
print "- - -"

### Using Shared Variables with 'updates'

x = T.fscalar('x')
state = theano.shared(0., 'state')
inc = theano.function([x], [state], updates=[(state, state+x)])

print "state value =", state.get_value()
inc(1); print "state value =", state.get_value()
inc(1); print "state value =", state.get_value()
inc(10); print "state value =", state.get_value()
state.set_value(0.); print "state value =", state.get_value()
print "- - -"

### Replace Shared Variables with 'givens'

x = T.fscalar('x')
z = state*2 + x
f = theano.function([x], z)  # using state internal value (0 at this time)

print "f(1) = state_value*2 + 1 =", f(1)  # 0*2 + 1
state.set_value(2); print "f(1) = state_value*2 + 1 =", f(1)  # 2*2 + 1

new_variable_that_replace_state = T.scalar(dtype=state.dtype, name='all_new')  # not a ?scalar, use dtype of state instead!
f = theano.function([new_variable_that_replace_state, x], z, givens=[(state, new_variable_that_replace_state)])

print "f(0,1) = 0*2 + 1 =", f(0,1)  # 0*2 + 1
print "f(2,1) = 2*2 + 1 =", f(2,1)  # 2*2 + 1
print "- - -"

### Copying functions with 'swap'

x = T.fscalar('x')
state = theano.shared(0., 'state')
inc = theano.function([x], state, updates=[(state, state+x)])

print "state value =", state.get_value()
inc(10); print "state value =", state.get_value()

new_state = theano.shared(0., 'new_state')
new_inc = inc.copy(swap={state:new_state})

print "new_state value =", new_state.get_value()
new_inc(100); print "new_state value =", new_state.get_value()
print "state value =", state.get_value()
print "- - -"

### Using Random Numbers

from theano.tensor.shared_randomstreams import RandomStreams

semi_random_number_generator = RandomStreams(234)
random_variable_uniform = semi_random_number_generator.uniform(size=(2,2))  # 2x2 matrix
random_variable_normal = semi_random_number_generator.normal((2,2))  # 2x2 matrix

f = theano.function([], random_variable_uniform)
# in g: not updating random_variable_normal so same number all the time
g = theano.function([], random_variable_normal, no_default_updates=True)
nearly_zeros = theano.function([], random_variable_uniform + random_variable_uniform - 2*random_variable_uniform)

print "f() ="; print f()  # this affects random_variable_uniform's generator
print "f() ="; print f()  # this affects random_variable_uniform's generator
print "g() ="; print g()  # this doesn't affects random_variable_normal's generator
print "g() ="; print g()  # this doesn't affects random_variable_normal's generator
# An important remark is that a random variable is drawn at most once during any single function execution.
# So nearly zero will be actual zeros (except for rounding errors)
print "nearly_zero ="; print nearly_zeros()  # this affects random_variable_uniform's generator
print "nearly_zero ="; print nearly_zeros()  # this affects random_variable_uniform's generator

random_generator = random_variable_uniform.rng.get_value(borrow=True)  # get the random generator for random variable uniform
random_generator.seed(934567)                                          # seeds the generator
random_variable_uniform.rng.set_value(random_generator, borrow=True)   # set back the seeded generator

semi_random_number_generator.seed(234)  # seed that will generate a random seed for each random variables
print "- - -"

### Computing Gradients

x = T.dscalar('x')
y = x**2
gy = T.grad(y, x)  # dy/dx = 2*x
theano.pp(gy)  # pretty print

f = theano.function([x], [gy])
print "f = grad(x^2). f(4) =", f(4)  # 8.0
print "- - -"

### Computing manually the Jacobian of y w.r.t. x with theano.scan()
# The term Jacobian designates the tensor comprising the first partial
# derivatives of the output of a function with respect to its inputs.

x = T.dvector('x')
y = x**2
J, updates = theano.scan(lambda i, y, x: T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])

f = theano.function([x], [J], updates=updates)
print "f = jacobian(x^2) = 2x. f([4,4]) ="; print f([4,4])  # array([[8.,0.],[0.,8.]])

# Can also do it automatically

J = theano.gradient.jacobian(y, x)
f = theano.function([x], [J])
print "f = jacobian(x^2) = 2x. f([4,4]) ="; print f([4,4])
print "- - -"

### Computing manually the Hessian of y w.r.t. x with T.grad() & theano.scan()
# The term Hessian designates the matrix comprising the second order
# partial derivative of a function with scalar output and vector input.

x = T.dvector('x')
y = x**2
cost = y.sum()

gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = theano.function([x], [H], updates=updates)
print "f = hessian(x^2) = grad(2x) = 2. f([4,4]) ="; print f([4,4])

# Can also do it automatically

H = theano.gradient.hessian(cost, x)
f = theano.function([x], [H])
print "f = hessian(x^2) = grad(2x) = 2. f([4,4]) ="; print f([4,4])
print "- - -"

### Jacobian times a Vector
# The R operator is built to evaluate the product between a Jacobian and a vector, namely f'(x) * v.

x = T.dvector('x')
W = T.dmatrix('W')
V = T.dmatrix('V')

y = T.dot(x, W)
JV = T.Rop(y, wrt=W, eval_points=V)
f = theano.function([x, W, V], [JV])

print "Jacobian * V =", f(x=[0, 1], W=[[1, 1], [1, 1]], V=[[2, 2], [2, 2]])  # [2, 2]
print "- - -"

### Vector times Jacobian
# The L-operator would compute a row vector times the Jacobian, namely v * f'(x).

x = T.dvector('x')
W = T.dmatrix('W')
V = T.dvector('V')

y = T.dot(x, W)
VJ = T.Lop(y, wrt=W, eval_points=V)
f = theano.function([x, W, V], [VJ], on_unused_input='warn')  # W is not used to compute f()!

print "V * Jacobian =", f(x=[0, 1], W=[[1, 1], [1, 1]], V=[2, 2])  # [[0,0],[2,2]]
print "- - -"

