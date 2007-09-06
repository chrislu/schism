
#ifndef CORE_SYSTEM_SINGLETON_H_INCLUDED
#define CORE_SYSTEM_SINGLETON_H_INCLUDED

#include <boost/noncopyable.hpp>

namespace scm {
namespace core {

class root_system;

template <class instance_t>
class core_system_singleton
{
private:
    // nested class definition
    template <class instance_t_>
    class core_system_singleton_ // : boost::noncopyable
    {
    public:
        explicit core_system_singleton_();
        virtual ~core_system_singleton_();

        instance_t_&                 get() const;
        instance_t_&                 operator*() const;
        instance_t_*const            get_ptr() const;
        instance_t_*const            operator->() const;

        void                         set_instance(instance_t_*const) const;
    private:
        mutable instance_t_*         _instance;

        // only system root has access to the write functions
        friend class scm::core::root_system;
    }; // class global_system_access_
public:
    typedef const core_system_singleton_<instance_t> type;
}; // struct global_system_access

} // namespace core
} // namespace scm

#include "core_system_singleton.inl"

#endif // CORE_SYSTEM_SINGLETON_H_INCLUDED
