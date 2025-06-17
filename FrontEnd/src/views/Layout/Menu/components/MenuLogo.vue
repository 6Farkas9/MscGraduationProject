<script setup lang="ts">
import { basic } from "@/config/setting"
import { useSettingStore } from "@/stores/modules/setting"
import { useRouter } from "vue-router"
const settingStore = useSettingStore()
const router = useRouter()
const pushHome = () => {
  router.push({
    path: "/",
  })
}
</script>

