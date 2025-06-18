<template>
  <div class="layout">
    <div
      class="layout-menu"
      :class="{
        'layout-min-menu': flag,
        bgMenuClassDark: dark,
      }"
    >
      <!-- 标题与logo -->
      <MenuLogo />
      <el-menu
        router
        :default-active="$route.path"
        :collapse="flag"
        :background-color="
          dark
            ? variables['menu-background-dark']
            : variables['menu-background']
        "
        text-color="#fff"
        :active-text-color="variables['menu-active-text']"
      >
        <MenuItem :menuRouteList="routerMenuList" />
      </el-menu>
    </div>
    <!-- 主体 -->
    <div
      class="layout-main"
      :class="{
        'layout-max-main': flag,
      }"
    >
      <!-- 头部 -->
      <div class="layout-header">
        <Header />
      </div>
      <!-- 导航栏 -->
      <TagViews />
      <!-- 主体部分 -->
      <div
        class="main-content"
        :style="{
          backgroundColor: dark ? '#1D1E1F' : '#fff',
          transition: 'all 0.2s',
        }"
      >
        <transition
          :enter-active-class="`animate__animated ${defaultSettings.routerAnimateInType} animate__faster`"
          mode="out-in"
        >
          <keep-alive>
            <router-view v-if="isflag" />
          </keep-alive>
        </transition>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import variables from "@/styles/variables.module.scss"
import { defaultSettings } from "@/config/setting"
import { useSettingStore } from "@/stores/modules/setting"

import { storeToRefs } from "pinia"
const settingStore = useSettingStore()

import { constRouter} from "@/router/route";
// 递归过滤路由，只保留需要显示的路由
const filterRoutes = (routes: any[]) => {
  return routes
    .filter(route => {
      // 如果是Layout组件，检查其children是否有需要显示的
      if (route.component?.name === 'Layout') {
        return route.children?.some(child => child.meta?.isShow)
      }
      // 普通路由直接检查isShow
      return route.meta?.isShow
    })
    .map(route => {
      // 如果是Layout组件，只保留需要显示的children
      if (route.component?.name === 'Layout') {
        return {
          ...route,
          children: filterRoutes(route.children || [])
        }
      }
      // 普通路由处理children（如果有）
      if (route.children) {
        return {
          ...route,
          children: filterRoutes(route.children)
        }
      }
      return route
    })
}

// 构造 routerMenuList
const routerMenuList = ref(filterRoutes(constRouter))

const { flag, dark, page_setting } = storeToRefs(settingStore)
let isflag = ref(true)
// 刷新
watch(
  () => settingStore.flush,
  () => {
    //让组件销毁
    isflag.value = false
    nextTick(() => {
      isflag.value = true
    })
  },
)
</script>

<style scoped lang="scss">
@import "./index.scss";
</style>
