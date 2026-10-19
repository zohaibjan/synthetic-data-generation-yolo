import {defineConfig} from 'vite';
export default defineConfig({base:'./',build:{outDir:'../synth/static',emptyOutDir:true},server:{proxy:{'/api':'http://127.0.0.1:8000'}},test:{environment:'jsdom'}});
