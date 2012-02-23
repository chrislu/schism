
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>
#include <limits>
#include <ostream>

#include <iostream>

#include <boost/io/ios_state.hpp>

namespace scm {
namespace data {

template<typename val_type, const unsigned bin_bits>
const scm::size_t
histogram_scalar<val_type, bin_bits>::signed_bin_offset = abs((std::numeric_limits<val_type>::min)() >> histogram_scalar<val_type, bin_bits>::value_bits);


template<typename val_type, const unsigned bin_bits>
histogram_scalar<val_type, bin_bits>::histogram_scalar()
  : _size(0)
{
    std::fill(_values, _values + bin_count, scm::size_t(0));
}

template<typename val_type, const unsigned bin_bits>
histogram_scalar<val_type, bin_bits>::~histogram_scalar()
{
}

template<typename val_type, const unsigned bin_bits>
void
histogram_scalar<val_type, bin_bits>::update(const val_type value)
{
    ++(_values[(value >> value_bits) + signed_bin_offset]);
    ++_size;
}

template<typename val_type, const unsigned bin_bits>
scm::size_t
histogram_scalar<val_type, bin_bits>::bin_value(const scm::size_t index) const
{
    assert(index < bin_count);
    return (_values[index]);
}

template<typename val_type, const unsigned bin_bits>
scm::size_t
histogram_scalar<val_type, bin_bits>::operator[](const scm::size_t index) const
{
    assert(index < bin_count);
    return (_values[index]);
}

template<typename val_type, const unsigned bin_bits>
scm::size_t
histogram_scalar<val_type, bin_bits>::size() const
{
    return (_size);
}

template<typename val_type, const unsigned bin_bits>
void
histogram_scalar<val_type, bin_bits>::clear()
{
    std::fill(_values, _values + bin_count, scm::size_t(0));
    _size = 0;
}

template<typename val_type, const unsigned bin_bits>
val_type
histogram_scalar<val_type, bin_bits>::low_value(const scm::size_t index) const
{
    return ((std::numeric_limits<val_type>::min)() + static_cast<val_type>(index * bin_value_count));
}

template<typename val_type, const unsigned bin_bits>
val_type
histogram_scalar<val_type, bin_bits>::high_value(const scm::size_t index) const
{
    return ((std::numeric_limits<val_type>::min)() + static_cast<val_type>((index + 1) * bin_value_count) - 1);
}

template<typename val_type, const unsigned bin_bits>
val_type
histogram_scalar<val_type, bin_bits>::center_value(const scm::size_t index) const
{
    return ((high_value(index) + low_value(index)) / 2);
}

template<typename val_type, const unsigned bin_bits>
void
histogram_scalar<val_type, bin_bits>::stream_out(std::ostream& os) const
{
    boost::io::ios_all_saver  ias(os);

    os << "size = " << size() << std::endl;
    os << "bin_count = " << bin_count << std::endl;
    //os << "value_bits = " << value_bits << std::endl;
    os << "bins = ";

    for (scm::size_t i = 0; i < bin_count; ++i) {
        os.width(4);
        os << std::endl << std::right << scm::int64(low_value(i)) << " - ";
        os.width(4);
        os << std::right << scm::int64(high_value(i)) << ": ";
        os.width(7);
        os << std::right << bin_value(i);
    }
}

template<typename val_type, const unsigned bin_bits>
std::ostream& operator<<(std::ostream& out_stream, const histogram_scalar<val_type, bin_bits>& out_hist)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_hist.stream_out(out_stream);

    return (out_stream);
}

} // namespace data
} // namespace scm
