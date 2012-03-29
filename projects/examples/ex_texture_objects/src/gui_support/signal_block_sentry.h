
#ifndef SCM_APP_GUI_SUPPORT_SIGNAL_BLOCK_SENTRY_H_INCLUDED
#define SCM_APP_GUI_SUPPORT_SIGNAL_BLOCK_SENTRY_H_INCLUDED

#include <QtCore/QObject>

namespace scm {
namespace gui {

class signal_block_sentry
{
public:
    signal_block_sentry(QObject*const object)
      : _object(object)
    {
        _signals_blocked = _object->signalsBlocked();
        _object->blockSignals(true);
    }
    ~signal_block_sentry()
    {
        restore();
    }
    void restore()
    {
        _object->blockSignals(_signals_blocked);
    }

private:
    QObject*const       _object;
    bool                _signals_blocked;
}; // class signal_block_sentry

} // namespace gui
} // namespace scm

#endif // SCM_APP_GUI_SUPPORT_SIGNAL_BLOCK_SENTRY_H_INCLUDED
