
#ifndef SCM_TIME_TIMER_HOST_H_INCLUDED
#define SCM_TIME_TIMER_HOST_H_INCLUDED

#include <set>
#include <string>

#include <boost/utility.hpp>

#include <scm/core/ptr_types.h>

namespace scm {
namespace time {

class timer_interface;

class timer_host : boost::noncopyable
{
public:
    typedef scm::core::shared_ptr<timer_interface>      timer_ptr;

    struct timer_info {
        std::string     _name;
        std::string     _description;
        timer_ptr       _timer;

        bool operator<(const timer_info& rhs) const {
            return (_name < rhs._name);
        }
    }; // struct timer_info

    typedef std::set<timer_info>                        timer_container;
    typedef timer_container::iterator                   iterator;
    typedef timer_container::const_iterator             const_iterator;

public:
    timer_host();
    virtual ~timer_host();

    void                add_timer(const std::string&    name,
                                  const std::string&    description,
                                  const timer_ptr&      timer);
    bool                remove_timer(const std::string& name);
    const timer_ptr&    get_timer(const std::string&    name) const;
    const timer_ptr&    operator[](const std::string&   name) const;

    const_iterator      begin() const;
    const_iterator      end() const;

protected:
    timer_container     _timers;

private:

}; // class timer_host

} // namespace time
} // namespace scm

#endif // SCM_TIME_TIMER_HOST_H_INCLUDED
