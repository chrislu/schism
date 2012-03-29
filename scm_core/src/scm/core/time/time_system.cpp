
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "time_system.h"

namespace scm {
namespace time {

ptime
local_time()
{
    return (boost::posix_time::microsec_clock::local_time());
}

ptime
universal_time()
{
    return (boost::posix_time::microsec_clock::universal_time());
}

date
local_date()
{
    return (boost::gregorian::day_clock::local_day());
}

date
universal_date()
{
    return (boost::gregorian::day_clock::universal_day());
}

} // namespace time
} // namespace scm
