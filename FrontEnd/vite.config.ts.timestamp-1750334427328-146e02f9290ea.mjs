// vite.config.ts
import { defineConfig, loadEnv } from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/vite@5.3.5_@types+node@20.14.12_sass@1.77.8/node_modules/vite/dist/node/index.js";
import { fileURLToPath, URL } from "node:url";
import UnoCSS from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/unocss@0.60.4_postcss@8.4.4_d77ecf598fb0b58f2ee868a183ae271b/node_modules/unocss/dist/vite.mjs";
import AutoImport from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/unplugin-auto-import@0.17.8_2f892aaa81c384b3d8117818b6f50ea2/node_modules/unplugin-auto-import/dist/vite.js";
import Components from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/unplugin-vue-components@0.2_899677613738026768c3cf909f9ba895/node_modules/unplugin-vue-components/dist/vite.js";
import { ElementPlusResolver } from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/unplugin-vue-components@0.2_899677613738026768c3cf909f9ba895/node_modules/unplugin-vue-components/dist/resolvers.js";
import { createSvgIconsPlugin } from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/vite-plugin-svg-icons@2.0.1_a02825df17793817b9416d7e20184bcf/node_modules/vite-plugin-svg-icons/dist/index.mjs";
import path from "path";
import vue from "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/node_modules/.pnpm/@vitejs+plugin-vue@5.1.1_vi_e64663e7138ac31b9ec3cf310910e21d/node_modules/@vitejs/plugin-vue/dist/index.mjs";
var __vite_injected_original_import_meta_url = "file:///D:/Desktop/GraduationDesign/GraduationDesign/FrontEnd/vite.config.ts";
var vite_config_default = defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  return {
    base: "/",
    plugins: [
      // ...
      AutoImport({
        imports: ["vue", "vue-router"],
        dts: "./src/types/auto-imports.d.ts",
        eslintrc: {
          enabled: true,
          // 指定文件存放目录
          filepath: "./.config/eslint-auto-import.json"
          // 自定义路径
        },
        vueTemplate: true,
        // default false
        resolvers: [
          ElementPlusResolver()
          // 自动导入图标组件
        ]
      }),
      vue(),
      UnoCSS(),
      // 主题定制
      Components({
        resolvers: [ElementPlusResolver({ importStyle: "sass" })],
        // 指定自定义组件位置(默认:src/components)自动注册全局组件
        dirs: [
          "src/components/ElementPlus_components/",
          "src/components/",
          "src/**/components"
        ],
        // 生成components.d.ts
        dts: "./src/types/components.d.ts",
        deep: true
      }),
      createSvgIconsPlugin({
        iconDirs: [path.resolve(process.cwd(), "src/assets/svgs")],
        symbolId: "icon-[dir]-[name]"
      })
    ],
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", __vite_injected_original_import_meta_url))
      }
    },
    //主题定制(主题覆盖)
    css: {
      preprocessorOptions: {
        scss: {
          javascriptEnabled: true,
          // 自动导入定制化样式文件进行样式覆盖
          additionalData: `  
          @use "@/styles/element/index.scss" as *;
           @use "@/config/public.scss" as *;
          `
        }
      }
    },
    //配置代理跨域
    server: {
      port: 9e3,
      //自定义端口
      proxy: {
        [env.VITE_BASE_URL]: {
          target: env.VITE_URL,
          changeOrigin: true,
          rewrite: (path2) => path2.replace(/^\/api/, "")
        }
      }
    }
  };
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJEOlxcXFxEZXNrdG9wXFxcXEdyYWR1YXRpb25EZXNpZ25cXFxcR3JhZHVhdGlvbkRlc2lnblxcXFxGcm9udEVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiRDpcXFxcRGVza3RvcFxcXFxHcmFkdWF0aW9uRGVzaWduXFxcXEdyYWR1YXRpb25EZXNpZ25cXFxcRnJvbnRFbmRcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0Q6L0Rlc2t0b3AvR3JhZHVhdGlvbkRlc2lnbi9HcmFkdWF0aW9uRGVzaWduL0Zyb250RW5kL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IHsgZGVmaW5lQ29uZmlnLCBsb2FkRW52IH0gZnJvbSBcInZpdGVcIlxyXG5pbXBvcnQgeyBmaWxlVVJMVG9QYXRoLCBVUkwgfSBmcm9tIFwibm9kZTp1cmxcIlxyXG5pbXBvcnQgVW5vQ1NTIGZyb20gXCJ1bm9jc3Mvdml0ZVwiXHJcbmltcG9ydCBBdXRvSW1wb3J0IGZyb20gXCJ1bnBsdWdpbi1hdXRvLWltcG9ydC92aXRlXCJcclxuaW1wb3J0IENvbXBvbmVudHMgZnJvbSBcInVucGx1Z2luLXZ1ZS1jb21wb25lbnRzL3ZpdGVcIlxyXG5pbXBvcnQgeyBFbGVtZW50UGx1c1Jlc29sdmVyIH0gZnJvbSBcInVucGx1Z2luLXZ1ZS1jb21wb25lbnRzL3Jlc29sdmVyc1wiXHJcbmltcG9ydCB7IGNyZWF0ZVN2Z0ljb25zUGx1Z2luIH0gZnJvbSBcInZpdGUtcGx1Z2luLXN2Zy1pY29uc1wiXHJcbmltcG9ydCBwYXRoIGZyb20gXCJwYXRoXCJcclxuaW1wb3J0IHsgdml0ZU1vY2tTZXJ2ZSB9IGZyb20gXCJ2aXRlLXBsdWdpbi1tb2NrXCJcclxuaW1wb3J0IHZ1ZSBmcm9tIFwiQHZpdGVqcy9wbHVnaW4tdnVlXCJcclxuLy8gaHR0cHM6Ly92aXRlanMuZGV2L2NvbmZpZy9cclxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKCh7IG1vZGUgfSkgPT4ge1xyXG4gIC8vXHU2MkZGXHU1MjMwXHU5MTREXHU3RjZFXHU3Njg0XHU3M0FGXHU1ODgzXHU1M0Q4XHU5MUNGXHJcbiAgY29uc3QgZW52ID0gbG9hZEVudihtb2RlLCBwcm9jZXNzLmN3ZCgpKVxyXG4gIHJldHVybiB7XHJcbiAgICBiYXNlOiBcIi9cIixcclxuICAgIHBsdWdpbnM6IFtcclxuICAgICAgLy8gLi4uXHJcbiAgICAgIEF1dG9JbXBvcnQoe1xyXG4gICAgICAgIGltcG9ydHM6IFtcInZ1ZVwiLCBcInZ1ZS1yb3V0ZXJcIl0sXHJcbiAgICAgICAgZHRzOiBcIi4vc3JjL3R5cGVzL2F1dG8taW1wb3J0cy5kLnRzXCIsXHJcbiAgICAgICAgZXNsaW50cmM6IHtcclxuICAgICAgICAgIGVuYWJsZWQ6IHRydWUsXHJcbiAgICAgICAgICAvLyBcdTYzMDdcdTVCOUFcdTY1ODdcdTRFRjZcdTVCNThcdTY1M0VcdTc2RUVcdTVGNTVcclxuICAgICAgICAgIGZpbGVwYXRoOiBcIi4vLmNvbmZpZy9lc2xpbnQtYXV0by1pbXBvcnQuanNvblwiLCAvLyBcdTgxRUFcdTVCOUFcdTRFNDlcdThERUZcdTVGODRcclxuICAgICAgICB9LFxyXG4gICAgICAgIHZ1ZVRlbXBsYXRlOiB0cnVlLCAvLyBkZWZhdWx0IGZhbHNlXHJcbiAgICAgICAgcmVzb2x2ZXJzOiBbXHJcbiAgICAgICAgICBFbGVtZW50UGx1c1Jlc29sdmVyKCksIC8vIFx1ODFFQVx1NTJBOFx1NUJGQ1x1NTE2NVx1NTZGRVx1NjgwN1x1N0VDNFx1NEVGNlxyXG4gICAgICAgIF0sXHJcbiAgICAgIH0pLFxyXG4gICAgICB2dWUoKSxcclxuICAgICAgVW5vQ1NTKCksXHJcblxyXG4gICAgICAvLyBcdTRFM0JcdTk4OThcdTVCOUFcdTUyMzZcclxuICAgICAgQ29tcG9uZW50cyh7XHJcbiAgICAgICAgcmVzb2x2ZXJzOiBbRWxlbWVudFBsdXNSZXNvbHZlcih7IGltcG9ydFN0eWxlOiBcInNhc3NcIiB9KV0sXHJcbiAgICAgICAgLy8gXHU2MzA3XHU1QjlBXHU4MUVBXHU1QjlBXHU0RTQ5XHU3RUM0XHU0RUY2XHU0RjREXHU3RjZFKFx1OUVEOFx1OEJBNDpzcmMvY29tcG9uZW50cylcdTgxRUFcdTUyQThcdTZDRThcdTUxOENcdTUxNjhcdTVDNDBcdTdFQzRcdTRFRjZcclxuICAgICAgICBkaXJzOiBbXHJcbiAgICAgICAgICBcInNyYy9jb21wb25lbnRzL0VsZW1lbnRQbHVzX2NvbXBvbmVudHMvXCIsXHJcbiAgICAgICAgICBcInNyYy9jb21wb25lbnRzL1wiLFxyXG4gICAgICAgICAgXCJzcmMvKiovY29tcG9uZW50c1wiLFxyXG4gICAgICAgIF0sXHJcbiAgICAgICAgLy8gXHU3NTFGXHU2MjEwY29tcG9uZW50cy5kLnRzXHJcbiAgICAgICAgZHRzOiBcIi4vc3JjL3R5cGVzL2NvbXBvbmVudHMuZC50c1wiLFxyXG4gICAgICAgIGRlZXA6IHRydWUsXHJcbiAgICAgIH0pLFxyXG4gICAgICBjcmVhdGVTdmdJY29uc1BsdWdpbih7XHJcbiAgICAgICAgaWNvbkRpcnM6IFtwYXRoLnJlc29sdmUocHJvY2Vzcy5jd2QoKSwgXCJzcmMvYXNzZXRzL3N2Z3NcIildLFxyXG4gICAgICAgIHN5bWJvbElkOiBcImljb24tW2Rpcl0tW25hbWVdXCIsXHJcbiAgICAgIH0pLFxyXG4gICAgXSxcclxuICAgIHJlc29sdmU6IHtcclxuICAgICAgYWxpYXM6IHtcclxuICAgICAgICBcIkBcIjogZmlsZVVSTFRvUGF0aChuZXcgVVJMKFwiLi9zcmNcIiwgaW1wb3J0Lm1ldGEudXJsKSksXHJcbiAgICAgIH0sXHJcbiAgICB9LFxyXG4gICAgLy9cdTRFM0JcdTk4OThcdTVCOUFcdTUyMzYoXHU0RTNCXHU5ODk4XHU4OTg2XHU3NkQ2KVxyXG4gICAgY3NzOiB7XHJcbiAgICAgIHByZXByb2Nlc3Nvck9wdGlvbnM6IHtcclxuICAgICAgICBzY3NzOiB7XHJcbiAgICAgICAgICBqYXZhc2NyaXB0RW5hYmxlZDogdHJ1ZSxcclxuICAgICAgICAgIC8vIFx1ODFFQVx1NTJBOFx1NUJGQ1x1NTE2NVx1NUI5QVx1NTIzNlx1NTMxNlx1NjgzN1x1NUYwRlx1NjU4N1x1NEVGNlx1OEZEQlx1ODg0Q1x1NjgzN1x1NUYwRlx1ODk4Nlx1NzZENlxyXG4gICAgICAgICAgYWRkaXRpb25hbERhdGE6IGAgIFxyXG4gICAgICAgICAgQHVzZSBcIkAvc3R5bGVzL2VsZW1lbnQvaW5kZXguc2Nzc1wiIGFzICo7XHJcbiAgICAgICAgICAgQHVzZSBcIkAvY29uZmlnL3B1YmxpYy5zY3NzXCIgYXMgKjtcclxuICAgICAgICAgIGAsXHJcbiAgICAgICAgfSxcclxuICAgICAgfSxcclxuICAgIH0sXHJcbiAgICAvL1x1OTE0RFx1N0Y2RVx1NEVFM1x1NzQwNlx1OERFOFx1NTdERlxyXG4gICAgc2VydmVyOiB7XHJcbiAgICAgIHBvcnQ6IDkwMDAsIC8vXHU4MUVBXHU1QjlBXHU0RTQ5XHU3QUVGXHU1M0UzXHJcbiAgICAgIHByb3h5OiB7XHJcbiAgICAgICAgW2Vudi5WSVRFX0JBU0VfVVJMXToge1xyXG4gICAgICAgICAgdGFyZ2V0OiBlbnYuVklURV9VUkwsXHJcbiAgICAgICAgICBjaGFuZ2VPcmlnaW46IHRydWUsXHJcbiAgICAgICAgICByZXdyaXRlOiAocGF0aCkgPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpLywgXCJcIiksXHJcbiAgICAgICAgfSxcclxuICAgICAgfSxcclxuICAgIH0sXHJcbiAgfVxyXG59KVxyXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQTJWLFNBQVMsY0FBYyxlQUFlO0FBQ2pZLFNBQVMsZUFBZSxXQUFXO0FBQ25DLE9BQU8sWUFBWTtBQUNuQixPQUFPLGdCQUFnQjtBQUN2QixPQUFPLGdCQUFnQjtBQUN2QixTQUFTLDJCQUEyQjtBQUNwQyxTQUFTLDRCQUE0QjtBQUNyQyxPQUFPLFVBQVU7QUFFakIsT0FBTyxTQUFTO0FBVDJNLElBQU0sMkNBQTJDO0FBVzVRLElBQU8sc0JBQVEsYUFBYSxDQUFDLEVBQUUsS0FBSyxNQUFNO0FBRXhDLFFBQU0sTUFBTSxRQUFRLE1BQU0sUUFBUSxJQUFJLENBQUM7QUFDdkMsU0FBTztBQUFBLElBQ0wsTUFBTTtBQUFBLElBQ04sU0FBUztBQUFBO0FBQUEsTUFFUCxXQUFXO0FBQUEsUUFDVCxTQUFTLENBQUMsT0FBTyxZQUFZO0FBQUEsUUFDN0IsS0FBSztBQUFBLFFBQ0wsVUFBVTtBQUFBLFVBQ1IsU0FBUztBQUFBO0FBQUEsVUFFVCxVQUFVO0FBQUE7QUFBQSxRQUNaO0FBQUEsUUFDQSxhQUFhO0FBQUE7QUFBQSxRQUNiLFdBQVc7QUFBQSxVQUNULG9CQUFvQjtBQUFBO0FBQUEsUUFDdEI7QUFBQSxNQUNGLENBQUM7QUFBQSxNQUNELElBQUk7QUFBQSxNQUNKLE9BQU87QUFBQTtBQUFBLE1BR1AsV0FBVztBQUFBLFFBQ1QsV0FBVyxDQUFDLG9CQUFvQixFQUFFLGFBQWEsT0FBTyxDQUFDLENBQUM7QUFBQTtBQUFBLFFBRXhELE1BQU07QUFBQSxVQUNKO0FBQUEsVUFDQTtBQUFBLFVBQ0E7QUFBQSxRQUNGO0FBQUE7QUFBQSxRQUVBLEtBQUs7QUFBQSxRQUNMLE1BQU07QUFBQSxNQUNSLENBQUM7QUFBQSxNQUNELHFCQUFxQjtBQUFBLFFBQ25CLFVBQVUsQ0FBQyxLQUFLLFFBQVEsUUFBUSxJQUFJLEdBQUcsaUJBQWlCLENBQUM7QUFBQSxRQUN6RCxVQUFVO0FBQUEsTUFDWixDQUFDO0FBQUEsSUFDSDtBQUFBLElBQ0EsU0FBUztBQUFBLE1BQ1AsT0FBTztBQUFBLFFBQ0wsS0FBSyxjQUFjLElBQUksSUFBSSxTQUFTLHdDQUFlLENBQUM7QUFBQSxNQUN0RDtBQUFBLElBQ0Y7QUFBQTtBQUFBLElBRUEsS0FBSztBQUFBLE1BQ0gscUJBQXFCO0FBQUEsUUFDbkIsTUFBTTtBQUFBLFVBQ0osbUJBQW1CO0FBQUE7QUFBQSxVQUVuQixnQkFBZ0I7QUFBQTtBQUFBO0FBQUE7QUFBQSxRQUlsQjtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUE7QUFBQSxJQUVBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQTtBQUFBLE1BQ04sT0FBTztBQUFBLFFBQ0wsQ0FBQyxJQUFJLGFBQWEsR0FBRztBQUFBLFVBQ25CLFFBQVEsSUFBSTtBQUFBLFVBQ1osY0FBYztBQUFBLFVBQ2QsU0FBUyxDQUFDQSxVQUFTQSxNQUFLLFFBQVEsVUFBVSxFQUFFO0FBQUEsUUFDOUM7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogWyJwYXRoIl0KfQo=
