import inspect

import pytest

from giving.utils import lax_function


def test_lax():
    @lax_function
    def f(x):
        return x * x

    assert f(10) == 100
    assert f(x=10, y=20) == 100


def test_lax_two_arguments():
    @lax_function
    def f(x, y):
        return x * y

    assert f(x=10, y=20) == 200
    assert f(x=10, y=20, z=30) == 200


def test_lax_closure():
    results = []

    @lax_function
    def f(x):
        oi = x * x
        results.append(oi)

    f(x=10, y=20)
    assert results == [100]


def test_lax_defaults():
    singleton = object()

    @lax_function
    def f(x=singleton):
        return x

    assert f() == singleton
    assert f(x=78) == 78
    assert f(y=78) == singleton


def test_lax_positional():
    @lax_function
    def f(x, /, y):
        return x + y

    assert f(1, 2) == 3
    with pytest.raises(TypeError):
        assert f(x=1, y=2) == 3
    assert f(1, y=2) == 3
    assert f(1, y=2, z=800) == 3
    assert f(1, y=2, x=800) == 3


def test_lax_positional_2():
    @lax_function
    def f(x, /):
        return x + 1

    assert f(2) == 3
    with pytest.raises(TypeError):
        assert f(x=2) == 3
    assert f(2) == 3
    assert f(2, z=800) == 3
    assert f(2, x=800) == 3


def test_lax_vararg():
    @lax_function
    def f(*xs):
        return sum(xs)

    assert f(2, 3, 4, z=8) == 9


def test_lax_generator():
    @lax_function
    def f(x):
        yield x
        yield x + 1

    assert inspect.isgeneratorfunction(f)
    assert list(f(5, z=9)) == [5, 6]


def test_lax_objcall():
    class Gorble:
        def __call__(self, x):
            return x * x

    f = lax_function(Gorble())

    assert f(x=4, y=6) == 16


def test_lax_already_kw():
    def f(**kws):
        return kws["x"]

    assert lax_function(f) is f
