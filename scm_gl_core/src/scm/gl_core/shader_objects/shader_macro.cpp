
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "shader_macro.h"

namespace scm {
namespace gl {

shader_macro::shader_macro()
{
}

shader_macro::shader_macro(const shader_macro& rhs)
  : _name(rhs._name)
  , _value(rhs._value)
{
}

shader_macro::shader_macro(const std::string& n, const std::string& v)
  : _name(n)
  , _value(v)
{
}

shader_macro_array
shader_macro::operator()(const std::string& n, const std::string& v)
{
    shader_macro_array ret(*this);

    return ret(n, v);
}

bool
shader_macro::operator==(const shader_macro& rhs) const
{
    return    (_name  == rhs._name)
           && (_value == rhs._value);
}

bool
shader_macro::operator!=(const shader_macro& rhs) const
{
    return    (_name  != rhs._name)
           || (_value != rhs._value);
}

shader_macro_array::shader_macro_array()
{
}

shader_macro_array::shader_macro_array(const shader_macro_array& rhs)
  : _array(rhs._array)
{
}

shader_macro_array::shader_macro_array(const shader_macro& in_macro)
  : _array(1, in_macro)
{
}

shader_macro_array::shader_macro_array(const std::string& n, const std::string& v)
  : _array(1, shader_macro(n, v))
{
}

shader_macro_array&
shader_macro_array::operator()(const shader_macro& in_macro)
{
    _array.push_back(in_macro);
    return (*this);
}

shader_macro_array&
shader_macro_array::operator()(const std::string& n, const std::string& v)
{
    _array.push_back(shader_macro(n, v));
    return (*this);
}

size_t
shader_macro_array::size() const
{
    return _array.size();
}

bool
shader_macro_array::empty() const
{
    return _array.empty();
}

const shader_macro_array::macro_vector&
shader_macro_array::macros() const
{
    return _array;
}

bool
shader_macro_array::operator==(const shader_macro_array& rhs) const
{
    return _array == rhs._array;
}

bool
shader_macro_array::operator!=(const shader_macro_array& rhs) const
{
    return _array != rhs._array;
}

} // namespace gl
} // namespace scm
