
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "vertex_format.h"

#include <cassert>

namespace scm {
namespace gl {

vertex_format::element::element(int                 stream,
                                const std::string&  name,
                                data_type           type,
                                int                 stride,
                                interger_handling   int_handling)
  : _buffer_stream(stream)
  , _attrib_name(name)
  , _attrib_location(-1)
  , _type(type)
  , _stride(stride)
  , _integer_handling(int_handling)
{
}

vertex_format::element::element(int               stream,
                                int               location,
                                data_type         type,
                                int               stride,
                                interger_handling int_handling)
  : _buffer_stream(stream)
  , _attrib_location(location)
  , _type(type)
  , _stride(stride)
  , _integer_handling(int_handling)
{
}

bool
vertex_format::element::operator==(const element& e) const
{
    return    (_buffer_stream == e._buffer_stream)
           && (_type == e._type)
           && (_stride == e._stride)
           && (_integer_handling == e._integer_handling)
           && (_attrib_location == e._attrib_location)
           && (_attrib_name == e._attrib_name);
}

bool
vertex_format::element::operator!=(const element& e) const
{
    return    (_buffer_stream != e._buffer_stream)
           || (_type != e._type)
           || (_stride != e._stride)
           || (_integer_handling != e._integer_handling)
           || (_attrib_location != e._attrib_location)
           || (_attrib_name != e._attrib_name);
}

vertex_format::vertex_format(const element_array& in_elements)
  : _elements(in_elements)
  , _generic(true)
{
    for (element_array::const_iterator i = in_elements.begin();
         (i != in_elements.end()) && _generic;
         ++i) {
        if (!i->_attrib_name.empty()) {
            _generic = false;
        }
    }
}

vertex_format::vertex_format(const element& in_element)
  : _elements(1, in_element)
  , _generic(true)
{
    if (!in_element._attrib_name.empty()) {
        _generic = false;
    }
}

vertex_format::vertex_format(int                stream,
                             const std::string& name,
                             data_type          type,
                             int                stride,
                             interger_handling  int_handling)
  : _elements(1, element(stream, name, type, stride, int_handling))
  , _generic(false)
{
}

vertex_format::vertex_format(int                stream,
                             int                location,
                             data_type          type,
                             int                stride,
                             interger_handling  int_handling)
  : _elements(1, element(stream, location, type, stride, int_handling))
  , _generic(true)
{
}

vertex_format::~vertex_format()
{
}

vertex_format&
vertex_format::operator()(const element& in_element)
{
    _elements.push_back(in_element);
    return (*this);
}

vertex_format&
vertex_format::operator()(int                stream,
                          const std::string& name,
                          data_type          type,
                          int                stride,
                          interger_handling  int_handling)
{
    _elements.push_back(element(stream, name, type, stride, int_handling));
    _generic = false;

    return *this;
}

vertex_format&
vertex_format::operator()(int               stream,
                          int               location,
                          data_type         type,
                          int               stride,
                          interger_handling int_handling)
{
    _elements.push_back(element(stream, location, type, stride, int_handling));
    _generic = true;

    return *this;
}

const vertex_format::element_array&
vertex_format::elements() const
{
    return (_elements);
}

bool
vertex_format::generic() const
{
    return (_generic);
}

bool
vertex_format::operator==(const vertex_format& rhs) const
{
    //if (_elements.size() != rhs._elements.size()) {
    //    return (false);
    //}
    //bool ret_value = true;
    //for (element_array::size_type i = 0; (i < _elements.size()) && ret_value; ++i) {
    //    ret_value = ret_value && (_elements[i] == rhs._elements[i]);
    //}
    //return (ret_value);
    return (_elements == rhs._elements);
}

bool
vertex_format::operator!=(const vertex_format& rhs) const
{
    //if (_elements.size() != rhs._elements.size()) {
    //    return (true);
    //}
    //bool ret_value = false;
    //for (element_array::size_type i = 0; (i < _elements.size()) && !ret_value; ++i) {
    //    ret_value = ret_value || (_elements[i] != rhs._elements[i]);
    //}
    //return (ret_value);
    return (_elements != rhs._elements);
}

} // namespace gl
} // namespace scm
