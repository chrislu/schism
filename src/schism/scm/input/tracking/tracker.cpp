
#include "tracker.h"

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
