<template>
  <div id="app">
    <router-view />
    <cache-viewer v-if="isCDNEnabled" :visible.sync="showCacheViewer" />
  </div>
</template>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

nav {
  padding: 30px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}

.copyright {
  text-align: center;
  color: rgb(0, 0, 0);
  font-size: 12px;
  font-weight: 400;
  margin-top: auto;
  padding: 30px 0 20px;
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 100%;
}

.el-message {
  top: 70px !important;
}
</style>
<script>
import CacheViewer from '@/components/CacheViewer.vue';
import { logCacheStatus } from '@/utils/cacheViewer';

export default {
  components: {
    CacheViewer
  },
  data() {
    return {
      showCacheViewer: false,
      isCDNEnabled: process.env.VUE_APP_USE_CDN === 'true'
    };
  },
  mounted() {
    // 只有在启用CDN时才添加相关事件和功能
    if (this.isCDNEnabled) {
      // 添加全局快捷键Alt+C用于显示缓存查看器
      document.addEventListener('keydown', this.handleKeyDown);

      // 在全局对象上添加缓存检查方法，便于调试
      window.checkCDNCacheStatus = () => {
        this.showCacheViewer = true;
      };
      
      // Console output for CDN cache tool
      console.info(
        '%c[Smart Service] CDN cache check tool loaded',
        'color: #409EFF; font-weight: bold;'
      );
      console.info(
        'Press Alt+C or run checkCDNCacheStatus() in the console to view CDN cache status'
      );

      // 检查Service Worker状态
      this.checkServiceWorkerStatus();
    } else {
      console.info(
        '%c[小智服务] CDN模式已禁用，使用本地打包资源',
        'color: #67C23A; font-weight: bold;'
      );
    }
  },
  beforeDestroy() {
    // 只有在启用CDN时才需要移除事件监听
    if (this.isCDNEnabled) {
      document.removeEventListener('keydown', this.handleKeyDown);
    }
  },
  methods: {
    handleKeyDown(e) {
      // Alt+C 快捷键
      if (e.altKey && e.key === 'c') {
        this.showCacheViewer = true;
      }
    },
    async checkServiceWorkerStatus() {
      // 检查Service Worker是否已注册
      if ('serviceWorker' in navigator) {
        try {
          const registrations = await navigator.serviceWorker.getRegistrations();
          if (registrations.length > 0) {
            console.info(
              '%c[Smart Service] Service Worker registered',
              'color: #67C23A; font-weight: bold;'
            );
            // Output cache status to console
            setTimeout(async () => {
              const hasCaches = await logCacheStatus();
              if (!hasCaches) {
                console.info(
                  '%c[Smart Service] No cache detected yet, please refresh or wait for cache to build',
                  'color: #E6A23C; font-weight: bold;'
                );
                // Extra tips for development
                if (process.env.NODE_ENV === 'development') {
                  console.info(
                    '%c[Smart Service] In development, Service Worker may not initialize cache properly',
                    'color: #E6A23C; font-weight: bold;'
                  );
                  console.info('Try the following to check if Service Worker is working:');
                  console.info('1. Check Service Worker status in DevTools > Application > Service Workers');
                  console.info('2. Check cache content in DevTools > Application > Cache > Cache Storage');
                  console.info('3. Use production build (npm run build) and access via HTTP server to test full functionality');
                }
              }
            }, 2000);
          } else {
            console.info(
              '%c[Smart Service] Service Worker not registered, CDN resources may not be cached',
              'color: #F56C6C; font-weight: bold;'
            );
            if (process.env.NODE_ENV === 'development') {
              console.info(
                '%c[Smart Service] This is normal in development',
                'color: #E6A23C; font-weight: bold;'
              );
              console.info('Service Worker usually only works in production');
              console.info('To test Service Worker functionality:');
              console.info('1. Run npm run build to build production version');
              console.info('2. Access the built page via HTTP server');
            }
          }
        } catch (error) {
          console.error('Failed to check Service Worker status:', error);
        }
      } else {
        console.warn('Current browser does not support Service Worker, CDN cache unavailable');
      }
    }
  }
};
</script>