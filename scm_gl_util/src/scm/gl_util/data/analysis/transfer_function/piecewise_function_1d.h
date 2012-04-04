
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_H_INCLUDED
#define SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_H_INCLUDED

#include <map>

#include <boost/static_assert.hpp>
#include <boost/call_traits.hpp>

namespace scm {
namespace data {
    
template<typename val_type,
         typename res_type>
class piecewise_function_1d
{
protected:
    typedef std::map<val_type, res_type, std::less<val_type> >  function_point_container_t;

public:
    typedef val_type                                            value_type;
    typedef res_type                                            result_type;
    typedef std::pair<val_type, res_type>                       stop_type;
    typedef typename function_point_container_t::iterator       stop_iterator;
    typedef typename function_point_container_t::const_iterator const_stop_iterator;
    typedef std::pair<stop_iterator, bool>                      insert_return_type;

public:
    piecewise_function_1d();
    piecewise_function_1d(const piecewise_function_1d<val_type, res_type>& ref) : _function(ref._function), _dirty(ref._dirty) {}
    piecewise_function_1d<val_type, res_type>& operator=(const piecewise_function_1d<val_type, res_type>& rhs) {
        _function = rhs._function;
        _dirty    = true;//rhs._dirty;
        return (*this);
    }

    insert_return_type      add_stop(const stop_type& stop);
    void                    del_stop(const stop_type& stop);
    insert_return_type      add_stop(value_type point, result_type value);
    void                    del_stop(value_type point);

    bool                    dirty() const;
    void                    dirty(const bool d);

    void                    clear();
    bool                    empty() const;

    result_type             operator[](float point) const;

    std::size_t             num_stops() const;

    stop_iterator           stops_begin();
    stop_iterator           stops_end();
    const_stop_iterator     stops_begin() const;
    const_stop_iterator     stops_end() const;

    stop_iterator           find_stop(val_type point);

    stop_iterator           find_lesser_stop(val_type point);
    stop_iterator           find_lequal_stop(val_type point);

    stop_iterator           find_gequal_stop(val_type point);
    stop_iterator           find_greater_stop(val_type point);

    const_stop_iterator     find_lesser_stop(val_type point) const;
    const_stop_iterator     find_lequal_stop(val_type point) const;

    const_stop_iterator     find_gequal_stop(val_type point) const;
    const_stop_iterator     find_greater_stop(val_type point) const;

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
    bool                                    _dirty;

private:
    //BOOST_STATIC_ASSERT(std::numeric_limits<val_type>::is_specialized);
    //BOOST_STATIC_ASSERT(std::numeric_limits<res_type>::is_specialized);
}; // class piecewise_function_1d

} // namespace data
} // namespace scm

#include "piecewise_function_1d.inl"

#endif // SCM_GL_UTIL_PIECEWISE_FUNCTION_1D_H_INCLUDED
