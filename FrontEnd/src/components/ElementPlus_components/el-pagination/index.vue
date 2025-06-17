<template>
  <el-pagination
    v-model:current-page="paginationParams.page"
    v-model:page-size="paginationParams.pageSize"
    :background="config.background"
    :disabled="config.disabled"
    :small="config.small"
    :layout="config.layout"
    :total="paginationParams.total"
    :page-sizes="config.page_sizes"
    @size-change="handleSizeChange"
    @current-change="handleCurrentChange"
  />
</template>

<script setup lang="ts">
import type { paginationConfigType, paginationParamsType } from "./type"
defineProps<{
  config: paginationConfigType
  paginationParams: paginationParamsType
}>()
const emit = defineEmits<{
  pageChange: [num: number]
  pageSizeChange: [num: number]
}>()
// 页码改变
const handleCurrentChange = (val: number) => {
  emit("pageChange", val)
}
// 页数改变
const handleSizeChange = (val: number) => {
  emit("pageSizeChange", val)
}
</script>
