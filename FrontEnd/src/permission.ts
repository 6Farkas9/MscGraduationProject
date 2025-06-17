// 路由基础配置
import nprogress from "./utils/nprogress";
import { basic } from "./config/setting";
import router from "@/router";

// 全局前置守卫
router.beforeEach((to, from, next) => {
  nprogress.start(); // 开启进度条
  document.title = `${basic.title}-${to.meta.title}`; // 设置页面标题
  next(); // 直接放行所有路由
});

// 全局后置守卫
router.afterEach(() => {
  nprogress.done(); // 关闭进度条
});