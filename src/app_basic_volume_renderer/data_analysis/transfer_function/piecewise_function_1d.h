
#ifndef SCALAR_TRANSFER_FUNCTION_1D_H_INCLUDED
#define SCALAR_TRANSFER_FUNCTION_1D_H_INCLUDED

#include <limits>
#include <map>
#include <set>

#include <boost/static_assert.hpp>

namespace scm
{
    template<typename val_type, typename scal_type>
    class piecewise_function_1d
    {
    protected:
        typedef std::map<val_type, scal_type, std::less<val_type> > function_point_container_t;

    public:
        typedef val_type                                            point_type;
        typedef scal_type                                           value_type;
        typedef typename function_point_container_t::const_iterator const_point_iterator;

    public:
        piecewise_function_1d();

        void                    add_point(val_type point, scal_type value, val_type epsilon = val_type(0));
        void                    del_point(val_type point, val_type epsilon = val_type(0));

        void                    clear();

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
    }; // class piecewise_function_1d



} // namespace scm

#include "piecewise_function_1d.inl"

#endif // SCALAR_TRANSFER_FUNCTION_1D_H_INCLUDED



