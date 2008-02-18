
namespace scm {
namespace math {

//template<typename scm_scalar, unsigned dim> 
//vec<scm_scalar, dim>::vec(){}
//
//template<typename scm_scalar, unsigned dim>
//template<typename scal_type>
//vec<scm_scalar, dim>::vec(const vec<scal_type, dim>& v)
//{
//    for(unsigned i = 0; i < dim; ++i) {
//        vec_array[i] = v.vec_array[i];
//    }
//}
//
//template<typename scm_scalar, unsigned dim> 
//vec<scm_scalar, dim>::vec(const scm_scalar s)
//{
//    for(unsigned i = 0; i < dim; ++i) {
//        vec_array[i] = s;
//    }
//}
//
//template<typename scm_scalar, unsigned dim>
//template<typename scal_type>
//vec<scm_scalar, dim>& vec<scm_scalar, dim>::operator=(const vec<scal_type, dim>& rhs)
//{ 
//    for(unsigned i = 0; i < dim; ++i) {
//        vec_array[i] = rhs.vec_array[i];
//    }
//    return (*this);
//}


//template<typename scal_type_l, typename scal_type_r, unsigned dim>
//inline const vec<scal_type_l, dim> operator+(const vec<scal_type_l, dim>& lhs, const vec<scal_type_r, dim>& rhs)
//{
//    vec<scal_type_l, dim> tmp_ret;
//
//    for (int i = 0; i < dim; ++i) {
//        tmp_ret.vec_array[i] = lhs.vec_array[i] + rhs.vec_array[i];
//    }
//
//    return (tmp_ret);
//}

template<class expr_type_l,
         class expr_type_r>
class vec_sum
{
    const expr_type_l   _l;
    const expr_type_r   _r;
public:
    vec_sum(const expr_type_l& l,
            const expr_type_r& r) : _l(l), _r(r) {}
    float operator[](int i) const {return (_l[i] + _r[i]); }
};

template<class    expr_type>
class vec_scal_prod
{
    float               _s;
    const expr_type     _e;
public:
    vec_scal_prod(const float s,
                  const expr_type e) : _s(s), _e(e) {}
    float operator[](int i) const {return (_s * _e[i]); }
};

template<class expr_type_l,
         class expr_type_r>
class vec_vec_prod
{
    const expr_type_l   _l;
    const expr_type_r   _r;
public:
    vec_vec_prod(const expr_type_l& l,
                 const expr_type_r& r) : _l(l), _r(r) {}
    float operator[](int i) const {return (_l[i] * _r[i]); }
};

template<class T>
class vec_expr
{
    const T     _expr;
public:
    vec_expr(const T& expr) : _expr(expr) {}
    float operator[](int i) const {return (_expr[i]); }
};

template<>
class vec_expr<vec4f>
{
    const vec4f&    _expr;
public:
    vec_expr(const vec4f& expr) : _expr(expr) {}
    float operator[](int i) const {return (_expr[i]); }
};

inline const vec_expr<vec_sum<vec_expr<vec4f>,  vec_expr<vec4f> > >
    operator+(const vec4f&       a,
              const vec4f&       b)
{
    return vec_expr<vec_sum<vec_expr<vec4f>,  vec_expr<vec4f> > >(vec_sum<vec_expr<vec4f>, vec_expr<vec4f> >(a, b));
}

template<class A>
inline const vec_expr<vec_sum<vec_expr<A>, vec_expr<vec4f> > >
    operator+(const vec_expr<A>& a,
              const vec4f&       b)
{
    return vec_expr<vec_sum<vec_expr<A>, vec_expr<vec4f> > >(vec_sum<vec_expr<A>, vec_expr<vec4f> >(a, b));
}

template<class B>
inline const vec_expr<vec_sum<vec_expr<vec4f>, vec_expr<B> > >
    operator+(const vec4f&       a,
              const vec_expr<B>& b)
{
    return vec_expr<vec_sum<vec_expr<vec4f>, vec_expr<B> > >(vec_sum<vec_expr<vec4f>, vec_expr<B> >(a, b));
}

template<class A, class B>
inline const vec_expr<vec_sum<vec_expr<A>, vec_expr<B> > >
    operator+(const vec_expr<A>& a,
              const vec_expr<B>& b)
{
    return vec_expr<vec_sum<vec_expr<A>, vec_expr<B> > >(vec_sum<vec_expr<A>, vec_expr<B> >(a, b));
}

inline const vec_expr<vec_scal_prod<vec_expr<vec4f> > >
    operator*(const float s,
              const vec4f& v)
{
    return vec_expr<vec_scal_prod<vec_expr<vec4f> > >(vec_scal_prod<vec_expr<vec4f> >(s, v));
}

template<class A>
inline const vec_expr<vec_scal_prod<vec_expr<A> > >
    operator*(const float s,
              const vec_expr<A>& v)
{
    return vec_expr<vec_scal_prod<vec_expr<A> > >(vec_scal_prod<vec_expr<A> >(s, v));
}
#if 1
//template<typename scal_type_l, typename scal_type_r>
inline const vec4f_c operator+(const vec4f_c& lhs, const vec4f_c& rhs)
{
    return (vec4f_c(lhs.x + rhs.x,
                                lhs.y + rhs.y,
                                lhs.z + rhs.z,
                                lhs.w + rhs.w));
}

//template<typename scal_type_l, typename scal_type_r, unsigned dim>
//inline const vec<scal_type_l, dim> operator*(const vec<scal_type_l, dim>& lhs, const vec<scal_type_r, dim>& rhs)
//{
//    vec<scal_type_l, dim> tmp_ret;
//
//    for (int i = 0; i < dim; ++i) {
//        tmp_ret.vec_array[i] = lhs.vec_array[i] * rhs.vec_array[i];
//    }
//
//    return (tmp_ret);
//}

//template<typename scal_type_l, typename scal_type_r>
inline const vec4f_c operator*(const vec4f_c& lhs, const vec4f_c& rhs)
{
    return (vec4f_c(lhs.x * rhs.x,
                                lhs.y * rhs.y,
                                lhs.z * rhs.z,
                                lhs.w * rhs.w));
}

//template<typename scal_type_l, typename scal_type_r, unsigned dim>
//inline const vec<scal_type_l, dim> operator*(const vec<scal_type_l, dim>& lhs, const scal_type_r rhs)
//{
//    vec<scal_type_l, dim> tmp_ret;
//
//    for (unsigned i = 0; i < dim; ++i) {
//        tmp_ret.vec_array[i] = lhs.vec_array[i] * rhs;
//    }
//
//    return (tmp_ret);
//}

//template<typename scal_type_l, typename scal_type_r>
inline const vec4f_c operator*(const vec4f_c& lhs, const float rhs)
{
    return (vec4f_c(lhs.x * rhs,
                                lhs.y * rhs,
                                lhs.z * rhs,
                                lhs.w * rhs));
}

//template<typename scal_type_l, typename scal_type_r, unsigned dim>
//inline const vec<scal_type_r, dim> operator*(const scal_type_l lhs, const vec<scal_type_r, dim>& rhs)
//{
//    vec<scal_type_r, dim> tmp_ret;
//
//    for (unsigned i = 0; i < dim; ++i) {
//        tmp_ret.vec_array[i] = lhs * rhs.vec_array[i];
//    }
//
//    return (tmp_ret);
//}

//template<typename scal_type_l, typename scal_type_r>
inline const vec4f_c operator*(const float lhs, const vec4f_c& rhs)
{
    return (vec4f_c(lhs * rhs.x,
                                lhs * rhs.y,
                                lhs * rhs.z,
                                lhs * rhs.w));
}

#endif

} // namespace math
} // namespace scm

