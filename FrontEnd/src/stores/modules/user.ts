import { getUserInfoApi, loginApi } from "@/api/mock"
import router from "@/router"
import { anyRouter, asyncRouter, constRouter } from "@/router/route"
import { GETTOKEN, REMOVETOKEN, SETTOKEN } from "@/utils/local"
import cloneDeep from "lodash/cloneDeep"
import { defineStore } from "pinia"
import { nextTick, ref } from "vue"
import type { RouteLocationOptions } from "vue-router"
interface loginParams {
  username: string
  password: number
}
export const useUserStore = defineStore(
  "user",
  () => {
    /***
     * @登录
     */
    const token = ref("" || GETTOKEN())
    const userinfo = ref(<any>{})
    // 用户登录
    const loginUp = async (loginParams: loginParams) => {
      const res = await loginApi(loginParams)
      if (res.code === 200) {
        token.value = res.data.token
        SETTOKEN(res.data.token)
        await getUserInfo(token.value)
        return "登录成功"
      } else {
        return Promise.reject(new Error(res.message))
      }
    }
    /**
     * @路由
     */

    // 路由表
    const routerMenuList = ref(constRouter as RouteLocationOptions[])
    // // 权限路过滤
    const filterRoute = (routes: any, asyncRoute: any) => {
      return asyncRoute.filter((route: any) => {
        if (routes.includes(route.name)) {
          if (route.children && route.children.length > 0) {
            route.children = filterRoute(routes, route.children)
          }
          return true
        }
      })
    }
    // 获取用户信息
    // 所有异步路由
    const getUserInfo = async (token: string) => {
      const res = await getUserInfoApi(token)
      if (res.code === 200) {
        userinfo.value = res.data
        // 异步路由
        const lastAsyncRoute = filterRoute(
          res.data.routes,
          cloneDeep(asyncRouter),
        )
        //常量路由，异步路由，任意路由（路由表,侧边栏）
        routerMenuList.value = [...constRouter, ...lastAsyncRoute, ...anyRouter]

        // 异步路由+任意路由
        const asyncRoutes = [...lastAsyncRoute, ...anyRouter]
        // 添加路由
        asyncRoutes.forEach((item: any) => {
          router.addRoute(item)
        })
      } else {
        return Promise.reject(new Error(res.message))
      }
    }
    // 重置路由
    const resetRoutes = () => {
      //移除所有路由
      routerMenuList.value.forEach((route: any) => {
        router.removeRoute(route.name)
      })
    }

    // 退出登录
    const loginOut = () => {
      REMOVETOKEN()
      userinfo.value = {}
      routerMenuList.value = []
      resetRoutes()
    }

    return {
      token,
      routerMenuList,
      userinfo,
      loginUp,
      loginOut,
      getUserInfo,
      resetRoutes,
    }
  },
  // {
  //   persist: [
  //     {
  //       key: "userinfo",
  //       paths: ["userinfo"],
  //     },
  //   ],
  // },
)
