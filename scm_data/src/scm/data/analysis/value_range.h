
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_DATA_VALUE_RANGE_H_INCLUDED
#define SCM_DATA_VALUE_RANGE_H_INCLUDED

namespace scm {
namespace data {

template <typename value_type>
class value_range
{
public:
    value_range() : _min(value_type(0)), _max(value_type(0)) {}
    value_range(const value_range& vr) : _min(vr._min), _max(vr._max) {}
    explicit value_range(const value_type min, const value_type max) : _min(min), _max(max) {}

    value_range& operator = (const value_range& rhs) { _min = rhs._min; _max = rhs._max; return (*this); }

    const value_type    min() const                             { return (_min); }
    const value_type    max() const                             { return (_max); }
    const value_type    half_range() const                      { return ((_max - _min) / value_type(2)); }
    const value_type    range() const                           { return (_max - _min); }

    void                set_min(const value_type m)             { _min = m; }
    void                set_max(const value_type m)             { _max = m; }

    void                decide_and_set_min(const value_type m)  { _min = (m < _min) ? m : _min; }
    void                decide_and_set_max(const value_type m)  { _max = (m > _max) ? m : _max; }

private:
    value_type          _min;
    value_type          _max;
}; // class value_range

} // namespace data
} // namespace scm

#endif // SCM_DATA_VALUE_RANGE_H_INCLUDED
