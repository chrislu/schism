

#ifndef HIGH_RES_TIMER_H_INCLUDED
#define HIGH_RES_TIMER_H_INCLUDED

#include <scm/core/time/timer_interface.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export(core) high_res_timer : public timer_interface
{
public:
    high_res_timer();
    virtual ~high_res_timer();

    void                start();
    void                stop();

protected:

private:
    time_stamp          _start;

}; // class timer_interface

} // namespace core
} // namespace scm

#endif // HIGH_RES_TIMER_H_INCLUDED
