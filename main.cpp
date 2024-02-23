#include <iostream>

#include <sycl/sycl.hpp>

#include <ImfArray.h>
#include <ImfRgbaFile.h>

int main(int argc, char **argv) {
  sycl::queue q{sycl::aspect_selector(sycl::aspect::fp16)};

  Imf::Rgba *pixels = nullptr;
  int width = -1;
  int height = -1;
  try {
    Imf::RgbaInputFile file("StillLife.exr");
    const auto dw = file.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    pixels = new Imf::Rgba[width * height];

    file.setFrameBuffer(pixels, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
  } catch (const std::exception &e) {
    std::cerr << "error reading image file StillLife.exr:" << e.what()
              << std::endl;
    return 1;
  }

  /* SERIAL
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        auto p = pixels[i * width + j];
        p.r = 0;
        p.g = 0;
        pixels[i * width + j] = p;
      }
    }
  */

  {
    sycl::buffer<Imf::Rgba, 2> rawBuffer{pixels, sycl::range<2>(height, width)};
    auto half4Buffer =
        rawBuffer.reinterpret<sycl::half4>(sycl::range<2>(height, width));

    q.submit([&half4Buffer, width, height](sycl::handler &h) {
      auto image = half4Buffer.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(
          sycl::range<2>(height, width), [image](sycl::item<2> item) {
            const auto index = item.get_id();
            const auto range = item.get_range();
            const auto x = index.get(0) / static_cast<float>(range.get(0));
            const auto y = index.get(1) / static_cast<float>(range.get(1));
            const auto mx = sycl::step(.5f, x);
            const auto my = sycl::step(.5f, y);
            auto p = image[index];
            p.r() *= mx;
            p.g() *= my;
            p.b() = 0;
            image[index] = p;
          });
    });
  }

  try {
    Imf::RgbaOutputFile file("out.exr", width, height, Imf::WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, width);
    file.writePixels(height);
  } catch (const std::exception &e) {
    std::cerr << "Error writing image file out.exr:" << e.what() << std::endl;
    return 1;
  }

  delete[] pixels;

  return 0;
}
