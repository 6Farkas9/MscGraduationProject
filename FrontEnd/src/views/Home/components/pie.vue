<template>
  <div class="pie" ref="pieRef" />
</template>

<script setup lang="ts">
import * as echarts from "echarts"
import { nextTick, onMounted, ref } from "vue"
const pieRef = ref()
onMounted(() => {
  nextTick(() => {
    let myChart = echarts.init(pieRef.value)

    let option = {
      color: ["#132C61", "#00FFFF", "#F5DEB3", "#f73303", "#FFDB5C"],
      legend: {
        top: "bottom",
        textStyle: {
          color: "gray",
        },
        itemWidth: 16,
        itemHeight: 15,
      },
      series: [
        {
          name: "",
          type: "pie",
          radius: ["5%", "65%"],
          center: ["52%", "40%"],
          roseType: "area",
          // label: {
          //   color: "white",
          // },
          labelLine: {
            length: 15,
            length2: 10,
          },
          data: [
            { value: 48, name: "上海" },
            { value: 42, name: "北京" },
            { value: 38, name: "广东" },
            { value: 25, name: "长沙" },
            { value: 30, name: "武汉" },
          ],
        },
      ],
    }
    option && myChart.setOption(option)
    window.addEventListener("resize", () => {
      myChart.resize()
    })
  })
})
</script>

<style scoped></style>
