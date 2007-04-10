
namespace math
{
    vec::vec()
    {
    }

    vec::vec(const vec<scm_scalar, dim>& v)
    {
        for(unsigned i = 0; i < dim; i++) {
            vec_array[i] = v.vec_array[i];
        }
    }

    vec::vec(const scm_scalar s) {
        for(unsigned i = 0; i < dim; i++) {
            vec_array[i] = s;
        }
    }

    vec::vec<scm_scalar, dim>& operator=(const vec<scm_scalar, dim>& rhs)
    { 
        for(unsigned i = 0; i < dim; i++) {
            vec_array[i] = rhs.vec_array[i];
        }
        return (*this);
    }

    scm_scalar& vec::operator[](const unsigned i)
    {
        assert(i < dim);
        return (vec_array[i]);
    }
    
    const scm_scalar& vec::operator[](const unsigned i) const
    {
        assert(i < dim);
        return (vec_array[i]);
    }


} // namespace math



