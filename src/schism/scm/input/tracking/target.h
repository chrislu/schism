
#ifndef SCM_INPUT_TARGET_H_INCLUDED
#define SCM_INPUT_TARGET_H_INCLUDED

#include <cstddef>

#include <scm/core/math/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace inp {

class __scm_export(input) target
{
public:
    target(std::size_t /*id*/);
    target(const target& /*ref*/);
    virtual ~target();

    const target&               operator=(const target&  /*rhs*/);
    void                        swap(target& /*ref*/);

    std::size_t                 id() const;
    const math::mat4f_t&        transform() const;
    void                        transform(const math::mat4f_t& /*trans*/);

protected:
    std::size_t                 _id;
    math::mat4f_t               _transform;

private:

}; // class target

} // namespace inp
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#include "target.inl"

#endif // SCM_INPUT_TARGET_H_INCLUDED
