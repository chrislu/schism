
#include "system_info.h"

namespace scm
{
    bool is_host_little_endian()
    {
        // based on wikipedia example code ;)
        int i = 1;
        char *p = (char *)&i;

        if (p[0] == 1) { // lowest address contains the lsb
            return (true);
        }
        else {
            return (false);
        }
    }

} // namespace scm


