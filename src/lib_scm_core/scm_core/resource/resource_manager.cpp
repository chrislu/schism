
#include "resource_manager.h"

using namespace scm::res;

resource_manager_base::resource_manager_base()
{
    _this.reset(this);
}

resource_manager_base::~resource_manager_base()
{
}
