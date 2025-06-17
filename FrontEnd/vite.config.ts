import { defineConfig, loadEnv } from "vite"
import { fileURLToPath, URL } from "node:url"
import UnoCSS from "unocss/vite"
import AutoImport from "unplugin-auto-import/vite"
import Components from "unplugin-vue-components/vite"
import { ElementPlusResolver } from "unplugin-vue-components/resolvers"
import { createSvgIconsPlugin } from "vite-plugin-svg-icons"
import path from "path"
import { viteMockServe } from "vite-plugin-mock"
import vue from "@vitejs/plugin-vue"
// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  //拿到配置的环境变量
  const env = loadEnv(mode, process.cwd())
  return {
    base: "./",
    plugins: [
      // ...
      AutoImport({
        imports: ["vue", "vue-router"],
        dts: "./src/types/auto-imports.d.ts",
        eslintrc: {
          enabled: true,
          // 指定文件存放目录
          filepath: "./.config/eslint-auto-import.json", // 自定义路径
        },
        vueTemplate: true, // default false
        resolvers: [
          ElementPlusResolver(), // 自动导入图标组件
        ],
      }),
      vue(),
      UnoCSS(),

      //mock
      viteMockServe({
        mockPath: "./src/mock",
        // localEnabled: true,
      }),
      // 主题定制
      Components({
        resolvers: [ElementPlusResolver({ importStyle: "sass" })],
        // 指定自定义组件位置(默认:src/components)自动注册全局组件
        dirs: [
          "src/components/ElementPlus_components/",
          "src/components/",
          "src/**/components",
        ],
        // 生成components.d.ts
        dts: "./src/types/components.d.ts",
        deep: true,
      }),
      createSvgIconsPlugin({
        iconDirs: [path.resolve(process.cwd(), "src/assets/svgs")],
        symbolId: "icon-[dir]-[name]",
      }),
    ],
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
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
          `,
        },
      },
    },
    //配置代理跨域
    server: {
      port: 9000, //自定义端口
      proxy: {
        [env.VITE_BASE_URL]: {
          target: env.VITE_URL,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  }
})
