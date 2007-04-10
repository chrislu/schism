
namespace std
{
    template<> void swap(scm::core::cvar& lhs, scm::core::cvar& rhs) {
        lhs.swap(rhs);
    }
} // namespace std

namespace scm
{
    namespace core
    {
        inline bool operator==(const cvar& lhs, const cvar& rhs)
        {
            return (lhs.equals(rhs));
        }

        inline bool operator==(const cvar& lhs, cvar::number_t rhs)
        {
            return (lhs.get_number_value() == rhs);
        }

        inline bool operator==(const cvar& lhs, const std::string& rhs)
        {
            return (lhs.get_string_value() == rhs);
        }

        inline bool operator==(cvar::number_t lhs, const cvar& rhs)
        {
            return (lhs == rhs.get_number_value());
        }

        inline bool operator==(const std::string& lhs, const cvar& rhs)
        {
            return (lhs == rhs.get_string_value());
        }

    } // namespace core
} // namespace scm

