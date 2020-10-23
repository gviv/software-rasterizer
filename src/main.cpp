#include <voxium.h>

class App : public vx::BaseApp
{
public:
    void draw()
    {
        vx::text("Hi!", 0, 0, vx::Color::WHITE);
    }
};

int main()
{
    App app{};
    vx::run(app, 1024, 1024, 256, 256);

    return 0;
}
