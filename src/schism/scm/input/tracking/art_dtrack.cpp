
#include "art_dtrack.h"

#include <boost/numeric/conversion/cast.hpp>

#include <cassert>

#include <scm/console.h>

#include <scm/core/math/math.h>
#include <scm/core/utilities/foreach.h>

#include <scm/input/tracking/target.h>
#include <scm/input/tracking/detail/dtrack.h>

namespace {

const std::size_t   dtrack_default_udp_bufsize  = 10000;

} // namespace 

namespace scm {
namespace inp {

art_dtrack::art_dtrack(std::size_t listening_port,
                       std::size_t timeout)
  : tracker(std::string("art_dtrack")),
    _dtrack(new DTrack),
    _listening_port(listening_port),
    _timeout(timeout),
    _initialized(false)
{
}

art_dtrack::~art_dtrack()
{
    shutdown();
}

bool art_dtrack::initialize()
{
    assert(_dtrack);

    if (_initialized) {
        console.get() << con::log_level(con::warning)
                      << "art_dtrack::initialize(): "
                      << "allready initialized" << std::endl;
        return (true);
    }

    int                         error_dtrack = 0;
    dtrack_init_type            init_dtrack;

    // initialize init struct
    init_dtrack.udpport         = boost::numeric_cast<unsigned short>(_listening_port);
    init_dtrack.udptimeout_us   = boost::numeric_cast<unsigned long>(_timeout);
    init_dtrack.udpbufsize      = boost::numeric_cast<int>(dtrack_default_udp_bufsize);
    init_dtrack.remote_port     = 0;
    strcpy(init_dtrack.remote_ip, "");

    
    // try to initialize dtrack device
    error_dtrack = _dtrack->init(&init_dtrack);
    
    if (error_dtrack != DTRACK_ERR_NONE) {
        console.get() << con::log_level(con::error)
                      << "art_dtrack::initialize(): "
                      << "unable to initialize dtrack device (error: '" << error_dtrack << "')" << std::endl;

        return (false);
    }

    // try to enable cameras and calculation
    error_dtrack = _dtrack->send_udp_command(DTRACK_CMD_CAMERAS_AND_CALC_ON, 1);

    if (error_dtrack != DTRACK_ERR_NONE) {
        console.get() << con::log_level(con::warning)
                      << "art_dtrack::initialize(): "
                      << "unable to enable cameras and calculation (error: '" << error_dtrack << "')" << std::endl;

        //return (false);
    }

    _initialized = true;

    return (true);
}

bool art_dtrack::shutdown()
{
    if (!_initialized) {
        //console.get() << con::log_level(con::warning)
        //              << "art_dtrack::shutdown(): "
        //              << "not initialized initialized" << std::endl;
        return (true);
    }

    // try to shutdown dtrack device
    int error_dtrack = 0;
    
    error_dtrack = _dtrack->exit();
    
    if (error_dtrack != DTRACK_ERR_NONE) {
        console.get() << con::log_level(con::error)
                      << "art_dtrack::shutdown(): "
                      << "unable to shutdown dtrack device (error: '" << error_dtrack << "')" << std::endl;

        return (false);
    }

    _initialized = false;

    return (true);
}

void art_dtrack::update(target_container& targets)
{
    typedef target_container::value_type    val_type;

    int                 error_dtrack        = 0;
    unsigned long       frame_nr            = 0;
    double              time_stamp          = 0.;
    int                 num_cal_bodies      = 0;
    int                 num_tracked_bodies  = 0;
    unsigned            max_tracked_bodies  = 0;
    int                 dummy               = 0;

    boost::scoped_array<dtrack_body_type>   bodies;

    foreach (const val_type& tar, targets) {
        max_tracked_bodies = math::max(boost::numeric_cast<unsigned>(tar.first), max_tracked_bodies);
    }

    bodies.reset(new dtrack_body_type[max_tracked_bodies]);

    // try to receive dtrack packet
    error_dtrack = _dtrack->receive_udp_ascii(  &frame_nr,              &time_stamp,    &num_cal_bodies,
                                                &num_tracked_bodies,    bodies.get(),   max_tracked_bodies,
                                                &dummy,                 0,              0,
                                                &dummy,                 0,              0,
                                                &dummy,                 0,              0);

    if (error_dtrack != DTRACK_ERR_NONE) {
        console.get() << con::log_level(con::warning)
                      << "art_dtrack::update(): "
                      << "unable to receive dtrack packet (error: '" << error_dtrack << "')" << std::endl;
        return;
    }

    const unsigned              max_bodies = math::min(max_tracked_bodies, boost::numeric_cast<unsigned>(num_tracked_bodies));
    target_container::iterator  target_it;

    for (unsigned i = 0; i < max_bodies; ++i) {
        target_it = targets.find(bodies[i].id + 1);
        //console.get() << con::log_level(con::error) << bodies[i].id << std::endl;

        if (target_it != targets.end()) {
            target_it->second.transform(math::mat4f_t(bodies[i].rot[0], bodies[i].rot[1], bodies[i].rot[2], 0.0f,   // 1st column
                                                      bodies[i].rot[3], bodies[i].rot[4], bodies[i].rot[5], 0.0f,   // 2nd column
                                                      bodies[i].rot[6], bodies[i].rot[7], bodies[i].rot[8], 0.0f,   // 3rd column
                                                      bodies[i].loc[0], bodies[i].loc[1], bodies[i].loc[2], 1.0f)); // 4th column
        }
    }
}

} // namespace inp
} // namespace scm
