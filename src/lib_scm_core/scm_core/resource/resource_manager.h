
#ifndef RESOURCE_MANAGER_H_INCLUDED
#define RESOURCE_MANAGER_H_INCLUDED

#include <map>

#include <boost/shared_ptr.hpp>

#include <scm_core/core/basic_system_interfaces.h>
#include <scm_core/core/ptr_types.h>
#include <scm_core/resource/resource.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace res {

class __scm_export resource_manager_base : public core::system
{
public:
    resource_manager_base();
    virtual ~resource_manager_base();

    using core::system::initialize;
    using core::system::shutdown;

private:
    virtual void               clear_instances()    = 0;

    boost::shared_ptr<resource_manager_base> _this;
}; // class resource_manager_base


template<class res_type>
class resource_manager : public resource_manager_base
{
public:
    typedef typename res_type::descriptor_type   res_desc_type;

protected:
    typedef core::shared_ptr<res_type>          res_ptr_type;
    typedef std::map<res_desc_type,
                     std::pair<res_ptr_type,
                               std::size_t> >   instance_container;

public:
    resource_manager();
    virtual ~resource_manager();

    bool                        initialize();
    bool                        shutdown();

    resource<res_type>          find_instance(const res_desc_type&          /*desc*/);
    resource<res_type>          create_instance(const res_desc_type&        /*desc*/);

    void                        register_instance(const resource<res_type>& /*inst*/);
    void                        release_instance(const resource<res_type>&  /*inst*/);

private:
    void                        clear_instances();
    instance_container          _loaded_instances;

    boost::shared_ptr<resource_manager<res_type> > _this;

}; // class resource_manager

} // namespace res
} // namespace scm

#include "resource_manager.inl"

#include <scm_core/utilities/platform_warning_enable.h>

#endif // RESOURCE_MANAGER_H_INCLUDED
