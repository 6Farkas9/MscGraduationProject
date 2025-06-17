<template>
  <!-- 中文 -->
  <el-config-provider :locale="zhCn">
    <router-view></router-view>
  </el-config-provider>
</template>
<script lang="ts" setup>
import { ElConfigProvider } from "element-plus"
import zhCn from "element-plus/es/locale/lang/zh-cn"
import { GETDARK } from "./utils/local"
onMounted(() => {
  if (GETDARK()) {
    document.documentElement.className = "dark"
  } else {
    document.documentElement.className = ""
  }
})
</script>
<style></style>
