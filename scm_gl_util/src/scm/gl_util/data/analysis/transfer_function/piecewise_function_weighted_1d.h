
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_PIECEWISE_FUNCTION_WEIGHTED_1D_H_INCLUDED
#define SCM_GL_UTIL_PIECEWISE_FUNCTION_WEIGHTED_1D_H_INCLUDED

#include <limits>
#include <map>
#include <set>

#include <boost/static_assert.hpp>
#include <boost/call_traits.hpp>

namespace scm {
namespace data {

template<typename val_type, typename scal_type>
class piecewise_function_weighted_1d
{
protected:
    struct weighted_value
    {
        weighted_value();
        weighted_value(typename boost::call_traits<scal_type>::param_type val, float wgt) : _value(val), _weight(wgt) {}
        weighted_value(const weighted_value& val) : _value(val._value), _weight(val._weight) {}
        weighted_value& operator=(const weighted_value& rhs) {
            _value = rhs._value;
            _weight = rhs._weight;
            return (*this);
        }

        scal_type   _value;
        float       _weight;
    };
    typedef std::map<val_type, weighted_value, std::less<val_type> > function_point_container_t;

public:
    typedef val_type                                            point_type;
    typedef scal_type                                           value_type;
    typedef typename function_point_container_t::const_iterator const_point_iterator;

public:
    piecewise_function_weighted_1d();
    piecewise_function_weighted_1d(const piecewise_function_weighted_1d<val_type, scal_type>& ref) : _function(ref._function) {}
    piecewise_function_weighted_1d<val_type, scal_type>& operator=(const piecewise_function_weighted_1d<val_type, scal_type>& rhs) {
        _function = rhs._function;
        return (*this);
    }

    void                    add_point(val_type point, typename boost::call_traits<scal_type>::param_type value, float weight, val_type epsilon = val_type(0));
    void                    del_point(val_type point, val_type epsilon = val_type(0));
    
    float                   get_point_weight(val_type point, val_type epsilon = val_type(0));

    void                    clear();
    bool                    empty() const;

    scal_type               operator[](float point) const;

    unsigned                get_num_points() const;
    const_point_iterator    get_points_begin() const;
    const_point_iterator    get_points_end() const;

    typename function_point_container_t::iterator           find_point(val_type point, val_type epsilon);

    typename function_point_container_t::iterator           find_lesser_point(val_type point);
    typename function_point_container_t::iterator           find_lequal_point(val_type point);

    typename function_point_container_t::iterator           find_gequal_point(val_type point);
    typename function_point_container_t::iterator           find_greater_point(val_type point);

    typename function_point_container_t::const_iterator     find_lesser_point(val_type point) const;
    typename function_point_container_t::const_iterator     find_lequal_point(val_type point) const;

    typename function_point_container_t::const_iterator     find_gequal_point(val_type point) const;
    typename function_point_container_t::const_iterator     find_greater_point(val_type point) const;

protected:

    struct lesser_op
    {
        lesser_op(val_type ref) : _ref(ref) {}
        bool operator()(const typename function_point_container_t::value_type& rhs) { return (rhs.first < _ref); }
    private:
        val_type _ref;
    };

    struct lequal_op
    {
        lequal_op(val_type ref) : _ref(ref) {}
        bool operator()(const typename function_point_container_t::value_type& rhs) { return (rhs.first <= _ref); }
    private:
        val_type _ref;
    };

    struct gequal_op
    {
        gequal_op(val_type ref) : _ref(ref) {}
        bool operator()(const typename function_point_container_t::value_type& rhs) { return (rhs.first >= _ref); }
    private:
        val_type _ref;
    };

    struct greater_op
    {
        greater_op(val_type ref) : _ref(ref) {}
        bool operator()(const typename function_point_container_t::value_type& rhs) { return (rhs.first > _ref); }
    private:
        val_type _ref;
    };

protected:
    function_point_container_t              _function;


private:
    //BOOST_STATIC_ASSERT(std::numeric_limits<val_type>::is_specialized);
    //BOOST_STATIC_ASSERT(std::numeric_limits<scal_type>::is_specialized);
}; // class piecewise_function_weighted_1d



} // namespace data
} // namespace scm

#include "piecewise_function_weighted_1d.inl"

#endif // SCM_GL_UTIL_PIECEWISE_FUNCTION_WEIGHTED_1D_H_INCLUDED
