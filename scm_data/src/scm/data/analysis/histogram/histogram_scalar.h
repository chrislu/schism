
#ifndef SCM_DATA_HISTOGRAM_SCALAR_H_INCLUDED
#define SCM_DATA_HISTOGRAM_SCALAR_H_INCLUDED

#include <iosfwd>
#include <limits>

#include <boost/static_assert.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

namespace scm {
namespace data {

template<typename val_type, const unsigned bin_bits = 4>
class histogram_scalar
{
private:
    BOOST_STATIC_ASSERT(std::numeric_limits<val_type>::is_integer);

private:
    static const unsigned               value_bits              = sizeof(val_type) * 8 - bin_bits;
    static const scm::size_t            signed_bin_offset;

public:
    static const scm::size_t            bin_count               = scm::size_t(1) << bin_bits;
    static const scm::size_t            bin_value_count         = scm::size_t(1) << value_bits;

    typedef val_type                    value_type;


public:
    histogram_scalar();
    /*virtual*/ ~histogram_scalar();

    void                                update(const val_type value);
    scm::size_t                         bin_value(const scm::size_t index) const;
    scm::size_t                         operator[](const scm::size_t index) const;

    scm::size_t                         size() const;

    void                                clear();

    val_type                            low_value(const scm::size_t index) const;
    val_type                            high_value(const scm::size_t index) const;
    val_type                            center_value(const scm::size_t index) const;

    void                                stream_out(std::ostream& os) const;

private:
    scm::size_t                         _values[bin_count];
    scm::size_t                         _size;

}; // class histogram_scalar

template<typename val_type, const unsigned bin_bits>
std::ostream& operator<<(std::ostream& out_stream, const histogram_scalar<val_type, bin_bits>& out_hist);

} // namespace data
} // namespace scm

#include "histogram_scalar.inl"

#endif // SCM_DATA_HISTOGRAM_SCALAR_H_INCLUDED
