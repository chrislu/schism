
#include "art_dtrack.h"

#include <scm/input/tracking/target.h>
#include <scm/input/tracking/detail/dtrack.h>

namespace scm {
namespace inp {

art_dtrack::art_dtrack()
  : tracker(std::string("art_dtrack")),
    _dtrack(new DTrack),
    _listening_port(0),
    _timeout(0)
{
}

art_dtrack::~art_dtrack()
{
}

bool art_dtrack::initialize()
{
    return (false);
}

void update()
{
}

bool art_dtrack::shutdown()
{
    return (true);
}

} // namespace inp
} // namespace scm
