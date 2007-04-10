
#include <exception>
#include <stdexcept>

namespace scm
{
    template<typename val_type>
    bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_1d<unsigned char, val_type>& scal_trafu, unsigned size)
    {
        if (size < 1) {
            return (false);
        }

        try {
            dst.reset(new val_type[size]);
        }
        catch (std::bad_alloc&) {
            dst.reset();
            return (false);
        }

        float a;
        float step = 255.0f / float(size - 1);
        for (unsigned i = 0; i < size; i++) {
            a = float(i) * step;
            dst[i] = scal_trafu[a]; 
        }

        return (true);
    }

} // namespace scm



