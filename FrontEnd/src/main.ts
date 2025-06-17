import { createApp } from "vue"
import storePinia from "./stores/index"
import component from "@/components/index"
import App from "./App.vue"
import router from "./router"
import "element-plus/theme-chalk/dark/css-vars.css"
// main.js
import "virtual:uno.css"

import "@/styles/index.scss"
// svg需要的配置代码
import "virtual:svg-icons-register"
// 导入路由鉴权
import "./permission"
// 导入mock数据
import "./mock/index"

const app = createApp(App)
app.use(component)
app.use(storePinia)
app.use(router)
app.mount("#app")
