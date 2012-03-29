
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "iomanip.h"

#include <ostream>
#include <iomanip>

#include <boost/io/ios_state.hpp>

namespace scm {
namespace io {

std::string
data_size_unit_string(const data_size_unit du)
{
    std::string s;

    switch (du) {
        case Byte: s.assign("B");    break;
        case KiB:  s.assign("KiB");  break;
        case MiB:  s.assign("MiB");  break;
        case GiB:  s.assign("GiB");  break;
        case TiB:  s.assign("TiB");  break;
        default:   s.assign("unknown unit"); break;
    }

    return s;
}

data_size::data_size(size_t ds, int dp)
  : _data_size(ds)
  , _data_unit(Byte)
  , _dec_places(dp)
  , _dynamic_unit(true)
{
}

data_size::data_size(size_t ds, data_size_unit du, int dp)
  : _data_size(ds)
  , _data_unit(du)
  , _dec_places(dp)
  , _dynamic_unit(false)
{
}

__scm_export(core) std::ostream& operator<<(std::ostream& os, const data_size& ds)
{
    std::ostream::sentry const  out_sentry(os);

    if (os) {
        boost::io::ios_all_saver saved_state(os);

        data_size_unit u = io::Byte;
        double         s = static_cast<double>(ds._data_size);

        if (ds._dynamic_unit) {
            if (0 < ds._data_size) {
                if      (0 < (ds._data_size / (1024ll * 1024ll * 1024ll * 1024ll))) { u = io::TiB; }
                else if (0 < (ds._data_size / (1024ll * 1024ll * 1024ll)))          { u = io::GiB; }
                else if (0 < (ds._data_size / (1024ll * 1024ll)))                   { u = io::MiB; }
                else if (0 < (ds._data_size / (1024ll)))                            { u = io::KiB; }
            }
        }
        else {
            u = ds._data_unit;
        }
        switch (u) {
            case KiB:  s = s / (1024.0); break;
            case MiB:  s = s / (1024.0 * 1024.0); break;
            case GiB:  s = s / (1024.0 * 1024.0 * 1024.0); break;
            case TiB:  s = s / (1024.0 * 1024.0 * 1024.0 * 1024.0); break;
            case Byte:
            default:   break;
        }

        std::string us = data_size_unit_string(u);

        if (io::Byte == u) {
            os << ds._data_size << us;
        }
        else {
            os << std::fixed << std::setprecision(ds._dec_places)
               << s << us;
        }
    }
    else {
        os.setstate(std::ios_base::failbit);
    }

    return os;
}

} // namespace io
} // namespace scm
