
namespace scm {
namespace math {

template<typename scal_type, const unsigned dim>
inline scal_type length_sqr(const vec<scal_type, dim>& lhs)
{
    return (dot(lhs, lhs));
}

template<typename scal_type, const unsigned dim>
inline scal_type length(const vec<scal_type, dim>& lhs)
{
    return (std::sqrt(length_sqr(lhs)));
}

template<typename scal_type, unsigned dim>
inline const vec<scal_type, dim> normalize(const vec<scal_type, dim>& lhs)
{
    return (lhs / length(lhs));
}

} // namespace math
} // namespace scm
