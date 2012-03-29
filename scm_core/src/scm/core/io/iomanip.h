
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_IOMANIP_H_INCLUDED
#define SCM_CORE_IO_IOMANIP_H_INCLUDED

#include <iosfwd>
#include <string>

#include <scm/core/numeric_types.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

enum data_size_unit {
    Byte    = 0x00,
    KiB,
    MiB,
    GiB,
    TiB
}; // enum data_size_unit

std::string
__scm_export(core)
data_size_unit_string(const data_size_unit du);

struct __scm_export(core) data_size
{
    size_t              _data_size;
    data_size_unit      _data_unit;
    int                 _dec_places;
    bool                _dynamic_unit;

    data_size(size_t ds, int dp = 2);
    data_size(size_t ds, data_size_unit du, int dp = 2);
};

__scm_export(core)
std::ostream& operator<<(std::ostream& os, const data_size& ds);

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_IO_IOMANIP_H_INCLUDED
