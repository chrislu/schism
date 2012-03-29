
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "tracker.h"

#include <scm/input/tracking/target.h>

namespace scm {
namespace inp {

tracker::tracker(const std::string& name)
  : _name(name)
{
}

tracker::~tracker()
{
}

const std::string& tracker::name() const
{
    return (_name);
}

} // namespace inp
} // namespace scm
