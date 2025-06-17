<template>
  <div class="canvas_dome">
    <el-row gutter="50">
      <el-col :xs="24" :sm="24" :md="24" :lg="11">
        <canvas ref="canvas"></canvas>
      </el-col>
      <el-col :xs="24" :sm="24" :md="24" :lg="12">
        <div class="select_canvas">
          <div class="select_img">
            <div class="tit">选择图片</div>
            <div class="picList">
              <img
                :src="pic"
                :class="{ activeImg: i === imgIndex }"
                v-for="(pic, i) in picList"
                :key="i"
                @click="selectImg(pic, i)"
              />
            </div>
          </div>
          <div class="select_color">
            <div class="tit">选择背景颜色</div>
            <div class="colorList">
              <div
                class="colorItem"
                :class="{ activeColorItem: i === colorIndex }"
                @click="selectBg(color, i)"
                v-for="(color, i) in colorList"
                :key="i"
              >
                <div class="color" :style="{ backgroundColor: color }"></div>
              </div>
            </div>
          </div>
          <div class="selece_input">
            <div class="tit">请输入文案</div>
            <el-input
              v-model="input_val"
              maxlength="5"
              placeholder="请输入文案"
            />
          </div>
          <div class="btn">
            <el-button type="info" @click="outPic">导出图片</el-button>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import debounce from "lodash/debounce"
import { message } from "@/Hooks/Element-plus"
import cat1 from "../../assets/image/cat1.png"
import cat2 from "../../assets/image/cat2.png"
import dog1 from "../../assets/image/dog1.png"
import dog2 from "../../assets/image/dog2.png"
import canvaas_bg from "../../assets/image/canvas.webp"
const picList = [dog1, cat2, dog2, cat1] as string[]
const colorList = [
  "#fbd04f",
  "#a8a1dc",
  "#83b6da",
  "#92c7ba",
  "#f6a356 ",
  "#5e5f61",
  "#f98787 ",
] as string[]
// canvas实例
const canvas = ref()
// 画布
const ctx = ref()
// 初始化背景图片
const initCanvas_bg = async () => {
  await new Promise((resolve, reject) => {
    // 初始化画布
    ctx.value = canvas.value.getContext("2d")!
    const width = 700
    const height = 800
    canvas.value.width = width
    canvas.value.height = height
    const img = new Image()

    img.onload = () => {
      ctx.value.drawImage(img, 0, 0, canvas.value.width, canvas.value.height) // 绘制第一张图片
      resolve("ok")
    }
    img.src = canvaas_bg
    img.onerror = () => {
      reject("失败")
    }
  })
}
onMounted(async () => {
  try {
    // 初始化背景canvas
    await initCanvas_bg()
  } catch (err: any) {
    message("error", "初始化背景失败")
  }
})

// 选择图片
const pic_val = ref("")
const imgIndex = ref<number>()
const selectImg = async (pic: string, i: number) => {
  pic_val.value = pic
  imgIndex.value = i
}
// 选择背景色
const color_val = ref("")
const colorIndex = ref<number>()
const selectBg = async (color: string, i: number) => {
  color_val.value = color
  colorIndex.value = i
}
// 输入值改变
const input_val = ref("")

watch(
  [() => pic_val.value, () => color_val.value, () => input_val.value],
  debounce(async (newVal: any) => {
    try {
      // 初始化背景canvas
      await initCanvas_bg()
      let [newPic_val, newColor_val, newInut_val] = newVal
      // 先看背景颜色有没有
      if (newColor_val) {
        // 设置填充色
        ctx.value.fillStyle = newColor_val
        // 绘制正方形
        ctx.value.fillRect(99, 75, 342, 477)
        // 再看文本
        if (newInut_val) {
          ctx.value.fillStyle = "black"
          ctx.value.font = `35px 隶书`
          ctx.value.fillText("名：" + newInut_val, 190, 180)
        }
        // 先铺背景在画图片
        if (newPic_val) {
          const img = new Image()
          img.src = newPic_val
          img.onload = () => {
            ctx.value.drawImage(img, 100, 202, 350, 350) // 绘制选择的图片
          }
        }
      } else {
        // 再看文本
        if (newInut_val) {
          ctx.value.fillStyle = "black"
          ctx.value.font = `35px 隶书`
          ctx.value.fillText("名：" + newInut_val, 190, 180)
        }
        // 先铺背景在画图片
        if (newPic_val) {
          const img = new Image()
          img.src = newPic_val
          img.onload = () => {
            ctx.value.drawImage(img, 100, 202, 350, 350) // 绘制选择的图片
          }
        }
      }
    } catch (err: any) {
      message("error", "初始化背景失败")
    }
  }, 100),
)
// 导出图片
const outPic = () => {
  let base64 = canvas.value.toDataURL()
  let link = document.createElement("a")
  link.textContent = "download image"
  link.href = base64
  link.download = "mypainting.jpeg"
  link.click()
}
</script>

<style scoped lang="scss">
.canvas_dome {
  width: 100%;
  height: 100%;
  padding: 10px;

  canvas {
    width: 100%;
    height: 98%;
  }
  .select_canvas {
    width: 100%;
    height: 100%;
    border: 1px gray solid;
    border-radius: 5px;
    padding: 5px;
    .select_img {
      cursor: pointer;
      margin-top: 10px;
      .tit {
        font-size: 15px;
        padding-bottom: 10px;
        font-weight: bold;
        color: gray;
        border-bottom: 1px var(--el-border-color-dark) solid;
      }
      .picList {
        display: flex;
        align-items: center;
        justify-content: space-around;
        flex-wrap: wrap;
        img {
          width: 120px;
          margin: 5px;
          border-radius: 10px;
          height: 120px;
          &.activeImg {
            // border: 2px var(--el-border-color-dark) solid;
            filter: drop-shadow(5px 5px 5px rgba(0, 0, 0, 0.5));
          }
        }
      }
    }
    .select_color {
      height: 20%;
      cursor: pointer;
      margin-top: 60px;
      .tit {
        font-size: 15px;
        font-weight: bold;
        padding-bottom: 10px;
        color: gray;
        border-bottom: 1px var(--el-border-color-dark) solid;
      }
      .colorList {
        display: flex;
        align-items: center;
        padding: 5px;
        .colorItem {
          margin: 0px 5px;
          width: 50px;
          height: 50px;
          display: flex;
          border-radius: 50%;
          &.activeColorItem {
            border: 2px var(--el-border-color-dark) solid;
          }
          .color {
            width: 38px;
            margin: auto;
            height: 38px;
            border-radius: 50%;
          }
        }
      }
    }
    .selece_input {
      cursor: pointer;
      margin-top: 50px;
      .tit {
        font-size: 15px;
        font-weight: bold;
        padding-bottom: 10px;
        color: gray;
        border-bottom: 1px var(--el-border-color-dark) solid;
      }
      .el-input {
        width: 95%;
        margin: 10px;
      }
    }
    .btn {
      width: 100%;
      display: flex;
      justify-content: flex-end;
      margin-top: 20px;
      margin-bottom: 50px;
      padding-right: 30px;
    }
  }
}
</style>
